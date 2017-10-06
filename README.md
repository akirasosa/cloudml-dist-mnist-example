# Cloud MLE and GCE compatible TensorFlow distributed training example

## Train the model in local

1. Create dataset.

```bash
python ./scripts/create_records.py 
```

Note: The dataset is stored in the [TFRecords][10] format.

[10]: https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details


2. Run ML locally.

```bash
gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --data_dir /tmp/data \
    --output_dir /tmp/ml-ouput \
    --train_steps 10000
```

## Train the model on Cloud Machine Learning

1. Create a bucket used for training jobs.

```bash
REGION=asia-east1
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT_ID}-ml2"
gsutil mkdir -c regional -l ${REGION} ${BUCKET}
```

2. Upload MNIST dataset to the training bucket.

```bash
python ./scripts/create_records.py 
gsutil cp /tmp/data/train.tfrecords ${BUCKET}/data/
gsutil cp /tmp/data/test.tfrecords ${BUCKET}/data/
```

3. Submit a training job to Cloud Machine Learning.

```bash
JOB_ID="${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training ${JOB_ID} \
  --package-path trainer \
  --module-name trainer.task \
  --staging-bucket ${BUCKET} \
  --job-dir ${BUCKET}/${JOB_ID} \
  --runtime-version 1.2 \
  --region ${REGION} \
  --config config/config.yaml \
  -- \
  --data_dir ${BUCKET}/data \
  --output_dir ${BUCKET}/${JOB_ID} \
  --train_steps 10000
```

Note: `JOB_ID` can be arbitrary, but you can't reuse the same one.

Note: Edit `config/config.yaml` to change the amount of resources
to be allocated for the job.

During the training, worker nodes show a training loss value (the total
loss value for dataset in a single training batch) in some intervals.
In addition, the master node shows a loss and accuracy for the testset
about every 3 minutes.

At the end of the training, the final evaluation against the testset is
shown as below. In this example, it achieved 99.3% accuracy for the testset.

```
Saving dict for global step 10008: accuracy = 0.9931, global_step = 10008, loss = 0.0315906
  ```

4. (Option) Visualize the training process with TensorBoard

After the training, the summary data is stored in
`${BUCKET}/${JOB_ID}` and you can visualize them with TensorBoard.
First, run the following command on the CloudShell to start TensorBoard.

```
$ tensorboard --port 8080 --logdir ${BUCKET}/${JOB_ID}
```

Select 'Preview on port 8080' from Web preview menu in the top-left corner
to open a new browser window:

![](docs/img/web-preview.png)

In the new window, you can use TensorBoard to see the training summary and
the visualized network graph, etc.

<img src="docs/img/tensorboard.png" width="600">

## Clean up

Clean up is really easy, but also super important: if you don't follow these
 instructions, you will continue to be billed for the project you created.

To clean up, navigate to the [Google Developers Console Project List][8],
 choose the project you created for this lab, and delete it. That's it.

[8]: https://console.developers.google.com/project
