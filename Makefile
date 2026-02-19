.PHONY: run debug install clean models face-model vosk-model

install:
	uv pip install -e .

run:
	.venv/bin/python main.py

debug:
	LOG_LEVEL=DEBUG .venv/bin/python main.py

clean:
	rm -rf __pycache__ *.egg-info .eggs

models: face-model vosk-model

face-model:
	@[ -f face_landmarker.task ] || \
	curl -L -o face_landmarker.task \
		"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

vosk-model:
	@[ -d vosk-model-small-en-us-0.15 ] || \
	(curl -L -o vosk-model.zip \
		"https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" && \
	unzip -q vosk-model.zip && \
	rm vosk-model.zip)
