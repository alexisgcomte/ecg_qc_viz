
FOLDER_PATH= .

run:
	. $(FOLDER_PATH)/env/bin/activate; \
	streamlit run streamlit_visualization/app.py\

download_streamlit_dataset_limited:
	. $(FOLDER_PATH)/env/bin/activate; \
	gdown https://drive.google.com/u/1/uc?id=1EH_RANmZTh03-oEDYu0cNaD3qRdLZdYQ&export=download; \

install_streamlit_dataset_limited:
	unzip df_ecg_103001_selection.zip -d dataset_streamlit/; \
	rm df_ecg_103001_selection.zip; \

download_streamlit_dataset_start:
	. $(FOLDER_PATH)/env/bin/activate; \
	gdown https://drive.google.com/u/1/uc?id=1Gr5lgwRW2NWvYDsMBEyagZys0hyGVVyN&export=download; \

install_streamlit_dataset_start:
	unzip df_ecg_130001_selection_start.zip -d dataset_streamlit/; \
	rm df_ecg_130001_selection_start.zip; \

download_streamlit_dataset_103001:
	. $(FOLDER_PATH)/env/bin/activate; \
	wget https://physionet.org/files/butqdb/1.0.0/103001/103001_ANN.csv -O dataset_streamlit/103001_ANN.csv; \
	gdown https://drive.google.com/u/1/uc?id=1-NqQkSJzNqTcG9bdKrBjtPsL1h-CLR8s&export=download; \

install_streamlit_dataset_103001:
	\
	mkdir dataset_streamlit; \
	unzip df_full_ecg_data_merge.zip -d dataset_streamlit/; \
	rm df_full_ecg_data_merge.zip; \
