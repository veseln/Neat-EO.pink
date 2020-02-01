import os
import re
import sys
import json
import collections

import numpy as np
from tqdm import tqdm

import psycopg2

from neat_eo.core import load_config, check_classes, make_palette, web_ui, Logs
from neat_eo.tiles import tiles_from_csv, tile_label_to_file, tile_bbox
from neat_eo.geojson import geojson_srid, geojson_tile_burn, geojson_parse_feature


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "rasterize", help="Rasterize GeoJSON or PostGIS features to tiles", formatter_class=formatter_class
    )

    inp = parser.add_argument_group("Inputs [either --sql or --geojson is required]")
    inp.add_argument("--cover", type=str, required=True, help="path to csv tiles cover file [required]")
    inp.add_argument("--config", type=str, help="path to config file [required, if no global config setting]")
    inp.add_argument("--type", type=str, required=True, help="type of features to rasterize (i.e class title) [required]")
    inp.add_argument("--geojson", type=str, nargs="+", help="path to GeoJSON features files")
    help = "SQL to retrieve geometry features [e.g SELECT geom FROM table WHERE ST_Intersects(TILE_GEOM, geom)]"
    inp.add_argument("--sql", type=str, help=help)
    inp.add_argument("--pg", type=str, help="If set, override config PostgreSQL dsn.")

    out = parser.add_argument_group("Outputs")
    out.add_argument("--out", type=str, required=True, help="output directory path [required]")
    out.add_argument("--append", action="store_true", help="Append to existing tile if any, useful to multiclass labels")
    out.add_argument("--ts", type=str, default="512,512", help="output tile size [default: 512,512]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def main(args):

    assert not (args.geojson is not None and args.pg is not None), "You have to choose between --pg or --geojson"
    assert len(args.ts.split(",")) == 2, "--ts expect width,height value (e.g 512,512)"

    config = load_config(args.config)
    check_classes(config)

    args.pg = config["auth"]["pg"] if not args.pg and "pg" in config["auth"].keys() else args.pg
    assert not (args.sql and not args.pg), "With --sql option, --pg dsn setting must also be provided"

    palette, transparency = make_palette([classe["color"] for classe in config["classes"]], complementary=True)
    index = [config["classes"].index(classe) for classe in config["classes"] if classe["title"] == args.type]
    assert index, "Requested type is not contains in your config file classes."
    burn_value = index[0]
    assert 0 < burn_value <= 255

    if args.sql:
        assert "limit" not in args.sql.lower(), "LIMIT is not supported"
        assert "TILE_GEOM" in args.sql, "TILE_GEOM filter not found in your SQL"
        sql = re.sub(r"ST_Intersects( )*\((.*)?TILE_GEOM(.*)?\)", "1=1", args.sql, re.I)
        assert sql and sql != args.sql, "Incorrect TILE_GEOM filter in your SQL"

    if os.path.dirname(os.path.expanduser(args.out)):
        os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    args.out = os.path.expanduser(args.out)
    log = Logs(os.path.join(args.out, "log"), out=sys.stderr)

    tiles = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))]
    assert len(tiles), "Empty Cover: {}".format(args.cover)

    if args.geojson:
        zoom = tiles[0].z
        assert not [tile for tile in tiles if tile.z != zoom], "Unsupported zoom mixed cover. Use PostGIS instead"

        feature_map = collections.defaultdict(list)

        log.log("neo rasterize - Compute spatial index")
        for geojson_file in args.geojson:

            with open(os.path.expanduser(geojson_file)) as geojson:
                feature_collection = json.load(geojson)
                srid = geojson_srid(feature_collection)

                for i, feature in enumerate(tqdm(feature_collection["features"], ascii=True, unit="feature")):
                    feature_map = geojson_parse_feature(zoom, srid, feature_map, feature)

        features = args.geojson

    if args.sql:
        conn = psycopg2.connect(args.pg)
        db = conn.cursor()

        db.execute("""SELECT ST_Srid("1") AS srid FROM ({} LIMIT 1) AS t("1")""".format(sql))
        srid = db.fetchone()[0]
        assert srid and int(srid) > 0, "Unable to retrieve geometry SRID."

        features = args.sql

    log.log("neo rasterize - rasterizing {} from {} on cover {}".format(args.type, features, args.cover))
    with open(os.path.join(os.path.expanduser(args.out), args.type.lower() + "_cover.csv"), mode="w") as cover:

        for tile in tqdm(tiles, ascii=True, unit="tile"):

            geojson = None

            if args.sql:
                w, s, e, n = tile_bbox(tile)
                tile_geom = "ST_Transform(ST_MakeEnvelope({},{},{},{}, 4326), {})".format(w, s, e, n, srid)

                query = """
                WITH
                  sql  AS ({}),
                  geom AS (SELECT "1" AS geom FROM sql AS t("1")),
                  json AS (SELECT '{{"type": "Feature", "geometry": '
                         || ST_AsGeoJSON((ST_Dump(ST_Transform(ST_Force2D(geom.geom), 4326))).geom, 6)
                         || '}}' AS features
                        FROM geom)
                SELECT '{{"type": "FeatureCollection", "features": [' || Array_To_String(array_agg(features), ',') || ']}}'
                FROM json
                """.format(
                    args.sql.replace("TILE_GEOM", tile_geom)
                )

                db.execute(query)
                row = db.fetchone()
                try:
                    geojson = json.loads(row[0])["features"] if row and row[0] else None
                except Exception:
                    log.log("Warning: Invalid geometries, skipping {}".format(tile))
                    conn = psycopg2.connect(args.pg)
                    db = conn.cursor()

            if args.geojson:
                geojson = feature_map[tile] if tile in feature_map else None

            if geojson:
                num = len(geojson)
                out = geojson_tile_burn(tile, geojson, 4326, list(map(int, args.ts.split(","))), burn_value)

            if not geojson or out is None:
                num = 0
                out = np.zeros(shape=list(map(int, args.ts.split(","))), dtype=np.uint8)

            tile_label_to_file(args.out, tile, palette, transparency, out, append=args.append)
            cover.write("{},{},{}  {}{}".format(tile.x, tile.y, tile.z, num, os.linesep))

    if not args.no_web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "."
        tiles = [tile for tile in tiles_from_csv(args.cover)]
        web_ui(args.out, base_url, tiles, tiles, "png", template)
