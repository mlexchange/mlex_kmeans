TAG 				:= latest	
USER 				:= mlexchange
PROJECT				:= k-means-dc

IMG_WEB_SVC    		:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP    	:= ${USER}/${PROJECT_JYP}:${TAG}
ID_USER				:= ${shell id -u}
ID_GROUP			:= ${shell id -g}

.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}
	echo ${ID_USER}

build_docker:
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile .

run_docker:
	docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ -v ${PWD}/../data:/app/data -p 8055:8055 ${IMG_WEB_SVC}

train_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python kmeans.py data/images/segment_series.tif data/model '{"n_clusters":2, "max_iter":300}'

test_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python segment.py data/images/segment_series.tif data/model/kmeans.joblib data/output '{"show_progress": 20}'

push_docker:
	docker push ${IMG_WEB_SVC}

clean:
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache
