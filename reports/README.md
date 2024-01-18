---
layout: default
nav_exclude: true
---

# Exam report for 02476 Machine Learning Operations

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**

7

### Question 2
> **Enter the study number for each member in the group**

s222977, s230250, s230251, s222948

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**

We used pytorch to train the Bert pre-trained network from hugging-face.

## Coding environment

> In the following section we are interested in learning more about your local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**

We used dvc to always fetch the same data file from the google drive. The data is a small csv file containing the text and label of around 6000 articles.
Docker is used to create a container to deploy our website.
docker build -f fastapi.dockerfile . -t fastapi:latest
docker run -p 8000:80 fastapi:latest


The project was implemented both in Google Cloud VM and both on the HPC of the DTU.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**

All in all we have covered most of the folders of the template excluding only the 'notebook' and the 'visualization' folder(which was replaced with the corresponding wandb folder).
In the 'src' folder, we have implemented most of our main code in corresponding subfolders including 'predict_model.py' and 'train_model.py'. Specifically, the first includes code relative to evaluation processes while the latter includes the training loop. The 'model.py' and config files were included in the 'models' subfolder and the 'make_dataset.py' in the 'data' subfolder.
In the rest of the folders, resulting models, logs, tests and texts can be found.

### Question 6
> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**

We followed PEP8 coding practice mostly on our src files covering areas such as indentation, line length, imports, whitespace, comments, and naming conventions, ensuring that Python code is readable and consistent. In addition, we used ruff as a linter to identify syntactical and stylish problems in our src files. By writing our code in this way we ensured that the quality and readability of the code will remain high throughout the whole process.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**

Yes, 5 for the data and 5 for the training 



### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**

Having coverage of 100% indicates a well-tested codebase, but it does not necessarily mean that the code is error-free since it only measures the percentage of code that is executed by the tests and it doesn't guarantee that every possible use case or input scenario has been tested.

### Question 9

> **Did your workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**

Yes, we had a repository on github and everyone had their own branch they worked on. Whenever anyone felt they had some working code that would improve the project, they merged it with the master branch.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**

We did, we had one csv file with text of around 6000 articles. It isn't a big file so the process was mostly for learning. We placed the csv file in the Google Drive and made sure the dvc config file pointed to the google drive url. It improved the workflow by always guaranteeing that the data is available. 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**

We ended up using pytest to perform unit tests to our codebase and ruff as a linter to highlight syntactical and stylistic problems. We performed several tests for the training process, fetching of the data and the creation of the datasets. We used caching for composing the various docker images.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**

Yes, hydra package was used and thus a config folder was made including different experiment yaml files along with a default one. Hyperparameters such as epochs, learning rate and scheduler_steps were included in those.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**

We used wandb to keep track of our experiments, plots etc. And hydra to keep track of our experimental params.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**

This image indicates the type of tests we did using Wandb [this figure](figures/wandb_team7.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**

We created three dockerfiles. One to run the same command as make train does, one to run the same command as make evaluate does and then we created a dockerfile that deploys our website with the availability of evaluation of text. run it by running docker run -p 8000:80 fastapi:latest after building it using docker build -f fastapi.dockerfile . -t fastapi:latest. The dockerfiles can be found in the dockerfiles folder.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**

While training the model, the profiler offered by pytorch was proven helpful enough to identify problems in the code and helped us optimize it. In addition, we used the debugger from visual studio code to insert breakpoints and examine certain behaviours. While testing, logging the errors helped us understand the reason the failures occured, but also pytest was proven helpful with its error logs. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
We mainly tried to understand the qualities of the GCP initially uploaded our data and project using Buckets from Cloud Storage which are mainly used to store data. Moreover, we used Compute Engine to create VMs both using our own docker image and using a provided one from GCP that includes necessary packages like torch. Additionally, we used Container Registry and tried to take advantage of Cloud Run in order to store and deploy our container images from the cloud.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
> 
By initializing instances in Compute Engine we were able to create VM's that we could access from whatever laptop/pc we might had. So instead of using our own machine we took advantage of the ones available in gcp. Even though we created both gpu and cpu based instances, we didn't end up using them for the most of the project.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
> 
Data of our project is here, [this figure](figures/buckert.png)


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Data of our project is here, [this figure](figures/containerr.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
> 
> Data of our project is here, [this figure](figures/cloud.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
> 
We were able to deploy our model both locally and in the hpc of DTU. Using hpc was easier for now and provided enough perks to make to procedure easier. Deploying the model in GCP was not accomplished because multiple errors could not be skipped.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
> 
No

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Around 30 dollars were spend, [this figure](figures/spent.png)


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Architecture [this figure](figures/overview_modded.png)



### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**

Understanding the benefits and qualities of all of the services we were taught about and being able to implement them in our project was challenging enough. Google cloud was one of the difficulties for sure since it includes the manipulation of different tools.
Additionally,The learning material on the slides proved very helpful for us along with searching stuff online and AI tools such as chatGPT. Overall although it was a struggle we consider this project one of the most complete and "professional" projects we have done so far.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**

s222977: I was responsible for DVC, Docker, and FastAPI, and kicked off the GitHub repository with Cookiecutter.
s230250: I was in charge of the correct data impelmentation/tokenization in order to feed the model and write unit tests for the dataset.
s230251: My responsibility was to write unit tests, ensure ruff is not complaining for our code and optimize the training process of the model.
s222948:

Everyone contributed to everything, with people being responsible for different parts of the project.
>>>>>>> 9a348b0162ab0f56881a8e7e5c15eef1eaaeb2c
