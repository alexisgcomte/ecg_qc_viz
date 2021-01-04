
FOLDER_PATH= .

run:
	. $(FOLDER_PATH)/env/bin/activate; \
	streamlit run streamlit_visualization/app.py\



download_streamlit_dataset_103001:
	. $(FOLDER_PATH)/env/bin/activate; \
	wget https://physionet.org/files/butqdb/1.0.0/103001/103001_ANN.csv -O dataset_streamlit/103001_ANN.csv; \
	gdown https://drive.google.com/u/1/uc?id=1-NqQkSJzNqTcG9bdKrBjtPsL1h-CLR8s&export=download; \

install_streamlit_dataset_103001:
	\
	mkdir dataset_streamlit; \
	unzip df_full_ecg_data_merge.zip -d dataset_streamlit/; \
	rm df_full_ecg_data_merge.zip; \



download_streamlit_dataset_111001:
	. $(FOLDER_PATH)/env/bin/activate; \
	wget https://physionet.org/files/butqdb/1.0.0/111001/111001_ANN.csv -O dataset_streamlit/111001_ANN.csv; \
	gdown https://drive.google.com/u/1/uc?id=1ybIrj8JaFpUYKWZgjbUIv4JGKIe5An1Q&export=download; \

install_streamlit_dataset_111001:
	\
	mkdir dataset_streamlit; \
	unzip df_full_ecg_data_merge_111001.zip -d dataset_streamlit/; \
	rm df_full_ecg_data_merge_111001.zip; \



download_streamlit_dataset_113001:
	. $(FOLDER_PATH)/env/bin/activate; \
	wget https://physionet.org/files/butqdb/1.0.0/113001/113001_ANN.csv -O dataset_streamlit/113001_ANN.csv; \
	gdown https://drive.google.com/u/1/uc?id=1-1EzbvjrRsaCqeBN2a9rIvNwNvN-vjKW&export=download; \

install_streamlit_dataset_113001:
	\
	mkdir dataset_streamlit; \
	unzip df_full_ecg_data_merge_113001.zip -d dataset_streamlit/; \
	rm df_full_ecg_data_merge_113001.zip; \



download_streamlit_dataset_124001:
	. $(FOLDER_PATH)/env/bin/activate; \
	wget https://physionet.org/files/butqdb/1.0.0/124001/124001_ANN.csv -O dataset_streamlit/124001_ANN.csv; \
	gdown https://drive.google.com/u/1/uc?id=1p3He2dup7uMbi0ftpPExHgrjI9t1NyUw&export=download; \

install_streamlit_dataset_124001:
	\
	mkdir dataset_streamlit; \
	unzip df_full_ecg_data_merge_124001.zip -d dataset_streamlit/; \
	rm df_full_ecg_data_merget_124001.zip; \