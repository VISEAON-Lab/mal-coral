##General utilities
from datetime import datetime
import time


##Upload predicted annotations to Labelbox project
def upload_to_labelbox(project, start_time, predictions):
    now = datetime.now()  # current date and time
    job_name = 'pre-labeling-' + str(now.strftime("%m-%d-%Y-%H-%M-%S"))

    upload_job = project.upload_annotations(
        name=job_name,
        annotations=predictions)

    print(upload_job)

    upload_job.wait_until_done()

    print("State", upload_job.state)

    print("Errors:", upload_job.errors) #new

    print("--- Finished in %s seconds ---" % (time.time() - start_time))
