# ML-model-development-example
Example of a dataset analysis and ML model development

1. Build Docker container:

    `docker build -t "ds_img"`

2. Run container:

    `docker run -it ds_img python3 run.py [Filename]`

    Example:

    `docker run -it ds_img python3 run.py dataset_00_with_header.csv`

    If filename is not set, default will be used (dataset_00_with_header.csv).

3. Running results on my server (Ubuntu 18.04, Xeon E3 8 cores 3.5 GHz, 32G RAM):

    ```df@ubuntu:~$ docker run -v /home/df/models/:/home/df/models -it ds_img python3 run.py dataset_00_with_header.csv train
    Overall fitting time: 297.054 sec.
    Model Performance:
        Train score: 0.999
        Test score: 0.946
	RMSE: 27.788
        Correct predictions: 17.080 %
        Average Error: 19.432
        Accuracy: 96.640
	```

    Note: Correct predictions metric is a number of predictions with an absolute error <= 3.
