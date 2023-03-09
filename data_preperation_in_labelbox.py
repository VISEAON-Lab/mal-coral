##General utilities
from multiprocessing.pool import ThreadPool
import os, os.path

#Local functions and definitions
from config import *
from utils import get_current_import_requests, delete_import_request, diff_lists, download_files, diff_lists_new

#Delete all existing bulk import requests from the Project. This will delete predictions from queued datarows.
def delete_labels_from_queued_datarows(client):
    all_import_requests = get_current_import_requests(client)

    for task in all_import_requests:
        response = delete_import_request(task['id'], client)
        print(response)

def compute_number_of_unlabeled_datarows(client, labels):
    ## Get datarows that needs to be pre-labeled. We are performing a subtraction (all datarows in project - labeled datarows)
    datarow_ids_with_labels = []

    for label in labels:
        datarow_ids_with_labels.append(label['DataRow ID'])

    all_datarow_ids = []
    all_datarows = []

    for dataset_id in DATASETS:
        dataset = client.get_dataset(dataset_id)
        for data_row in dataset.data_rows():
            all_datarow_ids.append(data_row.uid)
            all_datarows.append(data_row)

    datarow_ids_queued = diff_lists_new(all_datarow_ids, datarow_ids_with_labels)

    print('Number of all datarows: ', len(all_datarow_ids))
    print('Number of datarows to be pre-labeled: ', len(datarow_ids_queued))

    return all_datarows, all_datarow_ids, datarow_ids_queued

def download_unlabeled_datarows(all_datarows, datarow_ids_queued):
    ## Download queued datarows that needs to be pre-labeled

    data_row_queued = []
    data_row_queued_urls = []

    for datarow in all_datarows:
        for datarow_id in datarow_ids_queued:
            if datarow.uid == datarow_id:
                data_row_queued.append(datarow)
                extension = os.path.splitext(datarow.external_id)[1]
                filename = datarow.uid + extension
                data_row_queued_urls.append((DATA_LOCATION/inference/filename, datarow.row_data))

    print('Downloading queued data for inferencing...\n')
    filepath_inference = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, data_row_queued_urls)
    for item in filepath_inference:
        pass
    print('Success...\n')

    return data_row_queued
