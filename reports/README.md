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

s222977, s230250, s230251, 

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**

We used pytorch to train the Bert pre-trained network from hugging-face.

## Coding environment

> In the following section we are interested in learning more about your local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**

We used dvc to always fetch the same data file from the google drive. The data is a small csv file containing the text and label of around 6000 articles.
Docker is used to create a container of our environemnt both for training and evaluating. 
docker build -f trainer.dockerfile . -t trainer:latest
docker run trainer
docker build -f evaluate.dockerfile . -t evaluate:latest
docker run evaluate

The project was implemented both in Google Cloud VM and both on the HPC of the DTU.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
All in all we have covered most of the folders of the template excluding only the 'notebook' and the 'visualization' folder(which was replaced with the corresponding wandb folder).
In the 'src' folder, we have implemented most of our main code in corresponding subfolders including 'predict_model.py' and 'train_model.py'. Specifically, the first includes code relative to evaluation processes while the latter includes the training loop. The 'model.py' and config files were included in the 'models' subfolder and the 'make_dataset.py' in the 'data' subfolder.
In the rest of the folders, resulting models, logs, tests and texts can be found.

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**



### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**


### Question 9

> **Did your workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
??

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**


### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
Yes, hydra package was used and thus a config folder was made including different experiment yaml files along with a default one.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**



### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**



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
> Answer: We mainly tried to understand the qualities of the gcp cloud and thus we uploaded our data and project to gcp using 
Buckets from Cloud Storage. Moreover, we used Compute Engine to create VMs bot using our docker image and using a provided one from gcp inlcuding cuda support.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**


### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
We were able to deploy our model both locally and in the hpc of DTU. Using hpc was easier for now and provided enough perks to make to proceedure easier.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
Wandb??
### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**




### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
Understanding the benefits and qualities of all of the services we were taught about and being able to implement them in our project was challenging enough. Google cloud was one of the difficulties for sure since it includes the manipulation of different tools.
Additionally,.............

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**


>>>>>>> 9a348b0162ab0f56881a8e7e5c15eef1eaaeb2c
