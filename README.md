# SHIMR (Sparse High-order Interaction Model with Rejection option)
SHIMR is basically a forward feature selection with simultaneous sample reduction method to iteratively search for higher order feature interactions from the power set of complex features by maximizing the classification gain.

Sample reduction is achieved by incorporating the notion of "Classification with embedded reject option" which essentially minimizes the classification uncertainty, specifically in case of noisy data. One potential application of this method could be in clinical diagnosis (or prognosis). 

Below is a visualization of SHIMR when applied on "Breast Cancer Wisconsin (Diagnostic) Data Set" from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

<img src="Images/figure_RID_91550.png" width="800">



## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
## Prerequisites
<b> "SHIMR" </b> has the following two dependencies <br/>
1) CPLEX Optimizer  <br/>
2) Linear Time Closed Itemset Miner (LCM v5.3)  <br/>
&emsp; &ensp; Coded by Takeaki Uno,   e-mail:uno@nii.jp, 
homepage:   http://research.nii.ac.jp/~uno/code/lcm.html

Apart from that the current implementation is in python which is tested with the following python setup <br/>

1) Python 3.4.5 <br/>
2) scikit-learn==0.19.1 <br/>
3) scipy==1.0.0 <br/>
4) numpy==1.14.1 <br/>
5) pandas==0.22.0 <br/>
5) matplotlib==2.0.0 <br/>

### Download <br/>
<b> "IBM ILOG CPLEX Optimization Studio" </b>  from  https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ <br/>
<b> "LCM ver. 5.3" </b>  from  http://research.nii.ac.jp/~uno/codes.htm


## Installing
A step by step instructions that will guide you to get a working copy of "SHIMR" in your own development environment.

<b> A.  Create a virtual environment </b>

Download "anaconda" from https://www.continuum.io/downloads <br/>

1) Install Anaconda <br/>
&emsp; &ensp;  bash Anaconda-latest-Linux-x86_64.sh (Linux)  or <br/>
&emsp; &ensp;  bash Anaconda-latest-MacOSX-x86_64.sh (Mac) <br/>

2) Activate anaconda environment  <br/>
&emsp; &ensp; source anaconda/bin/activate anaconda/

3) Create a new environment and activate it <br/>
&emsp; &ensp; conda create -n r_boost python=3.4.5 <br/>
&emsp; &ensp; source activate r_boost <br/>
&emsp; &ensp; pip install -r requirements.txt <br/>


<b> B.  Install "IBM ILOG CPLEX Optimization Studio" </b>

1) Download "cplex_studioXXX.linux-x86.bin" (Linux) or "cplex_studioXXX.osx.bin" (Mac) file <br/>

Make sure the .bin file is executable. If necessary, change its permission using the chmod command from the directory where the .bin is located: <br/>
chmod +x cplex_studioXXX.linux-x86.bin. <br/>

2) Enter the following command to start the installation process: <br/>
./cplex_studioXXX.linux-x86.bin <br/>

3) Installation path: <br/>
/home/user/ibm/ILOG/CPLEX_StudioXXX <br/>
4) cd ‘/home/username/ibm2/ILOG/CPLEX_StudioXXX/cplex/python/3.4/x86-64_linux’ ==> Linux <br/>

or cd  /Users/username/Applications/IBM/ILOG/CPLEX_StudioXXX/cplex/python/3.4/x86-64_osx/ ==> Mac <br/>

5) python setup.py install <br/>

<b> C.  Install "LCM ver. 5.3" </b>

1) Unzip the 'lcm53.zip' directory <br/>
2) cd lcm53 <br/>
3) make. <br/>

## Running the tests
To test SHIMR we included "Breast Cancer Wisconsin (Diagnostic) Data Set" 
from UCI Machine Learning Repository under the Data folder.
Please run 'code/main_WDBC.ipynp' in an interactive mode to see the sparse high order interactions of features generated by
our visualization module.

## Visualization module
Motivation of our visualization module came from "UpSet: Visualizing Intersecting Sets" (http://caleydo.org/tools/upset/) and its python implementation (https://github.com/ImSoErgodic/py-upset).





