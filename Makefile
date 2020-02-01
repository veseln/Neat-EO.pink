help:
	@echo "This Makefile rules are designed for Neat-EO.pink devs and power-users."
	@echo "For plain user installation follow README.md instructions, instead."
	@echo ""
	@echo ""
	@echo " make install     To install, few Python dev tools and Neat-EO.pink in editable mode."
	@echo "                  So any further Neat-EO.pink Python code modification will be usable at once,"
	@echo "                  throught either neo tools commands or neat_eo.* modules."
	@echo ""
	@echo " make check       Launchs code tests, and tools doc updating."
	@echo "                  Do it, at least, before sending a Pull Request."
	@echo ""
	@echo " make check_tuto  Launchs neo commands embeded in tutorials, to be sure everything still up to date."
	@echo "                  Do it, at least, on each CLI modifications, and before a release."
	@echo "                  NOTA: It takes a while."
	@echo ""
	@echo " make pink        Python code beautifier,"
	@echo "                  as Pink is the new Black ^^"



# Dev install
install:
	pip3 install pytest black flake8 twine
	pip3 install -e .


# Lauch all tests
check: ut it doc
	@echo "==================================================================================="
	@echo "All tests passed !"
	@echo "==================================================================================="


# Python code beautifier
pink:
	black -l 125 *.py neat_eo/*.py neat_eo/*/*.py tests/*py tests/*/*.py


# Perform units tests, and linter checks
ut:
	@echo "==================================================================================="
	black -l 125 --check *.py neat_eo/*.py neat_eo/*/*.py
	@echo "==================================================================================="
	flake8 --max-line-length 125 --ignore=E203,E241,E226,E272,E261,E221,W503,E722
	@echo "==================================================================================="
	pytest tests -W ignore::UserWarning


# Launch Integration Tests
it: it_pre it_train it_post


# Integration Tests: Data Preparation
it_pre:
	@echo "==================================================================================="
	rm -rf it
	neo info
	neo cover --zoom 18 --bbox 4.8,45.7,4.82,45.72 --out it/cover
	neo download --rate 20 --type WMS --url "https://download.data.grandlyon.com/wms/grandlyon?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=Ortho2015_vue_ensemble_16cm_CC46&WIDTH=512&HEIGHT=512&CRS=EPSG:3857&BBOX={xmin},{ymin},{xmax},{ymax}&FORMAT=image/jpeg" --cover it/cover --out it/images
	echo "Download Buildings GeoJSON" && wget --show-progress -q -nc -O it/lyon_roofprint.json "https://download.data.grandlyon.com/wfs/grandlyon?SERVICE=WFS&REQUEST=GetFeature&TYPENAME=ms:fpc_fond_plan_communaut.fpctoit&VERSION=1.1.0&srsName=EPSG:4326&BBOX=4.79,45.69,4.83,45.73&outputFormat=application/json; subtype=geojson" | true
	echo "Download Roads GeoJSON" && wget --show-progress -q -nc -O it/lyon_road.json "https://download.data.grandlyon.com/wfs/grandlyon?SERVICE=WFS&VERSION=1.1.0&request=GetFeature&typename=pvo_patrimoine_voirie.pvochausseetrottoir&outputFormat=application/json; subtype=geojson&SRSNAME=EPSG:4326&bbox=`neo cover --dir it/images --type extent`" | true
	ogr2ogr -f SQLite it/lyon_road.sqlite it/lyon_road.json -dsco SPATIALITE=YES -t_srs EPSG:3857 -nln roads -lco GEOMETRY_NAME=geom
	ogr2ogr -f GeoJSON it/lyon_road_poly.json it/lyon_road.sqlite -dialect sqlite -sql "SELECT Buffer(geom, IFNULL(largeurchaussee, 5.0) / 2.0) AS geom FROM roads"
	neo rasterize --type Building --geojson it/lyon_roofprint.json --config config.toml --cover it/cover --out it/labels
	neo rasterize --type Road --geojson it/lyon_road_poly.json --config config.toml --cover it/cover --append --out it/labels
	neo rasterize --type Building --geojson it/lyon_roofprint.json --config config.toml --cover it/cover --out it/labels_osm
	neo cover --dir it/images --splits 80/20 --out it/train/cover it/eval/cover
	neo subset --dir it/images --cover it/train/cover --out it/train/images
	neo subset --dir it/labels --cover it/train/cover --out it/train/labels
	neo subset --dir it/images --cover it/eval/cover --out it/eval/images
	neo subset --dir it/labels --cover it/eval/cover --out it/eval/labels
	mkdir --parents it/predict/tiff
	wget -nc -O it/predict/tiff/1841_5174_08_CC46.tif "https://download.data.grandlyon.com/files/grandlyon/imagerie/ortho2018/ortho/GeoTiff_YcBcR/1km_8cm_CC46/1841_5174_08_CC46.tif"
	wget -nc -O it/predict/tiff/1842_5174_08_CC46.tif "https://download.data.grandlyon.com/files/grandlyon/imagerie/ortho2018/ortho/GeoTiff_YcBcR/1km_8cm_CC46/1842_5174_08_CC46.tif"
	neo tile --zoom 18 --rasters it/predict/tiff/*.tif --out it/predict/images
	neo cover --zoom 18 --dir it/predict/images --out it/predict/cover
	echo "Download PBF" && wget -nc -O it/predict/ra.pbf "http://download.geofabrik.de/europe/france/rhone-alpes-latest.osm.pbf" | true
	osmium extract --bbox `neo cover --dir it/predict/images --type extent` -o it/predict/lyon.pbf it/predict/ra.pbf
	neo extract --type Building --pbf it/predict/lyon.pbf --out it/predict/osm_building.json
	neo extract --type Road --pbf it/predict/lyon.pbf --out it/predict/osm_road.json
	neo rasterize --type Building --geojson it/predict/osm_building.json --config config.toml --cover it/predict/cover --out it/predict/labels
	neo rasterize --type Road --geojson it/predict/osm_road.json --config config.toml --cover it/predict/cover --append --out it/predict/labels



# Integration Tests: Training
it_train:
	@echo "==================================================================================="
	export CUDA_VISIBLE_DEVICES=0 && neo train --config config.toml --bs 4 --lr 0.00025 --epochs 2 --train_dataset it/train --classes_weights `neo dataset --mode weights --dataset it/train --config config.toml` --out it/pth
	export CUDA_VISIBLE_DEVICES=0,1 && neo train --config config.toml --bs 4 --lr 0.00025 --epochs 4 --resume --checkpoint it/pth/checkpoint-00002.pth --classes_weights auto --train_dataset it/train --eval_dataset it/eval --out it/pth
	export CUDA_VISIBLE_DEVICES=0,1 && neo train --config config.toml --bs 4 --optimizer AdamW --lr 0.00025 --epochs 6 --resume --checkpoint it/pth/checkpoint-00004.pth --classes_weights auto --train_dataset it/train --eval_dataset it/eval --out it/pth
	neo train --config config.toml --bs 4 --checkpoint it/pth/checkpoint-00006.pth --classes_weights auto --eval_dataset it/eval --out it/pth
	neo info --checkpoint it/pth/checkpoint-00006.pth


# Integration Tests: Post Training
it_post:
	@echo "==================================================================================="
	neo export --checkpoint it/pth/checkpoint-00006.pth --type jit --out it/pth/export.jit
	neo export --checkpoint it/pth/checkpoint-00006.pth --type onnx --out it/pth/export.onnx
	neo predict --config config.toml --bs 8 --checkpoint it/pth/checkpoint-00006.pth --dataset it/predict --out it/predict/masks
	neo predict --metatiles --config config.toml --bs 8 --checkpoint it/pth/checkpoint-00006.pth --dataset it/predict --out it/predict/masks_meta
	neo compare --config config.toml --images it/predict/images it/predict/labels it/predict/masks --mode stack --labels it/predict/labels --masks it/predict/masks_meta --out it/predict/compare
	neo compare --images it/predict/images it/predict/compare --mode side --out it/predict/compare_side
	neo compare --config config.toml --mode list --labels it/predict/labels --max Building QoD 0.50 --masks it/predict/masks_meta --geojson --out it/predict/compare/tiles.json
	cp it/predict/compare/tiles.json it/predict/compare_side/tiles.json
	neo vectorize --type Building --config config.toml --masks it/predict/masks_meta --out it/predict/building.json
	neo vectorize --type Road --config config.toml --masks it/predict/masks_meta --out it/predict/road.json


# Documentation generation (tools and config file)
doc:
	@echo "==================================================================================="
	@echo "# Neat-EO.pink tools documentation" > docs/tools.md
	@for tool in `ls neat_eo/tools/[^_]*py | sed -e 's#.*/##g' -e 's#.py##'`; do \
		echo "Doc generation: $$tool"; 						  \
		echo "## neo $$tool" >> docs/tools.md; 				  	  \
		echo '```'           >> docs/tools.md; 				  	  \
		neo $$tool -h        >> docs/tools.md; 				  	  \
		echo '```'           >> docs/tools.md; 				  	  \
	done
	@echo "Doc generation: config.toml"
	@echo "## config.toml"        > docs/config.md; 			  	  \
	echo '```'                    >> docs/config.md; 			  	  \
	cat config.toml               >> docs/config.md; 			  	  \
	echo '```'                    >> docs/config.md;
	@echo "Doc generation: Makefile"
	@echo "## Makefile"           > docs/makefile.md; 			  	  \
	echo '```'                    >> docs/makefile.md; 			  	  \
	make --no-print-directory     >> docs/makefile.md; 			  	  \
	echo '```'                    >> docs/makefile.md;


# Check neo commands embeded in Documentation
check_doc:
	@echo "==================================================================================="
	@echo "Checking README:"
	@echo "==================================================================================="
	@rm -rf ds && sed -n -e '/```bash/,/```/ p' README.md | sed -e '/```/d' > .CHECK && sh .CHECK
	@echo "==================================================================================="


# Check neo commands embeded in Tutorials
check_tuto: check_101

check_101:
	@echo "==================================================================================="
	@echo "Checking 101"
	@mkdir -p tuto && cd tuto && mkdir 101 && sed -n -e '/```bash/,/```/ p' ../docs/101.md | sed -e '/```/d' > 101/.CHECK && cd 101 && sh .CHECK && cd ..
	@cd tuto/101 && tar cf 101.tar ds/images ds/labels predict/images predict/osm predict/masks predict/compare predict/compare_side



# Send a release on PyPI
pypi:
	rm -rf dist Neat-EO.pink.egg-info
	python3 setup.py sdist
	twine upload dist/* -r pypi
