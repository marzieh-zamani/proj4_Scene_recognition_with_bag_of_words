# Project 4: [Scene recognition with bag of words]


# Project Review
## Project subject: 
Scene recognition with bag of words

## Project objectives:
- The goal of this project is to get familiar with image recognition.

## Steps to local feature matching between two images (image1 & image 2):
1. Represent images using:
=>> get_tiny_images()
=>> build_vocabulary()
=>> get_bags_of_sifts()

2. Classify images using: 
=>> nearest_neighbor_classify()
=>> svm_classify()


# Main files to check
- Project report: I have briefly introduced the objectives of the project, reviewed the image processing methods, explained the main functions, described experiments and discussed the results.

- Jupyter notebook: High level code where inputs are given, main functions are called, results are displayed and saved.

- Student code: Image processing functions are defined.


# Setup by Dr. Kin-Choong Yow
- Install [Miniconda](https://conda.io/miniconda). It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment using the given file by modifying the following command based on your OS (`linux`, `mac`, or `win`): `conda env create -f environment_<OS>.yml`
- This should create an environment named `ense885ay`. Activate it using the following Windows command: `activate ense885ay` or the following MacOS / Linux command: `source activate ense885ay`.
- Run the notebook using: `jupyter notebook ./code/proj3.ipynb`
- Generate the submission once you're finished using `python zip_submission.py`


# Credits and References
This project has been developed based on the project template and high-level code provided by Dr. Kin-Choong Yow, my instructor for the course “ENSE 885AY: Application of Deep Learning in Computer Vision”.

This course is based on Georgia Tech’s CS 6476 Computer Vision course instructed by James Hays.

- Dr. Kin-Choong Yow page: 
http://uregina.ca/~kyy349/

- “CS 6476 Computer Vision” page:
https://www.cc.gatech.edu/~hays/compvision/

- Project source page at “CS 6476 Computer Vision”:
Not found

- James Hays pages:
https://www.cc.gatech.edu/~hays/
https://github.com/James-Hays?tab=repositories


# My contribution
The following files contain the code written by me:
- code/student_code.py >> get_tiny_images() function
- code/student_code.py >> build_vocabulary() function
- code/student_code.py >> get_bags_of_sifts() function
- code/student_code.py >> nearest_neighbor_classify() function
- code/student_code.py >> svm_classify() function
- proj4_experiment.ipynb >> Experiment code for cross validation and validation 

`### TODO: YOUR CODE HERE ###`

`# My lines of code are inside these comments to be separated from the rest of the code.`

`### END OF STUDENT CODE ####`

______________
Marzieh Zamani