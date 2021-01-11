# ecg_qc_viz

# ecg_qc_viz

Patient used:
patient: 103001

## I - Installation / Prerequisites

#### Dependencies

ecg_qc requires:

- Python (>= 3.6)
- ecg-qc (>=1.0b3)
- plotly  (>=4.14.1)
- streamlit (>=0.74.1)


#### User installation

For manual install, use follow commands:

    $ git clone https://github.com/alexisgcomte/ecg_qc_viz.git
    $ cd ecg_qc_viz/
    $ python -m venv env && source env/bin/activate
    $ pip install -r requirements.txt


## II - How to use

0) Create dataset_streamlit folder
```bash
mkdir dataset_streamlit
```

1) Download the dataset
```bash
make download_streamlit_dataset_103001
```

2) Extract for streamlit 

```bash
make install_streamlit_dataset_103001
```

3) Run Streamlit

```bash
make run
```
