[![Generic badge](https://img.shields.io/github/license/marcortiz11/FastComposedModels)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/github/v/release/marcortiz11/FastComposedModels)](https://zenodo.org/badge/latestdoi/209519696)

# Fast Composed Models
**FCM (Fast Composing Models)** is a pure Python framework designed to efficiently build and evaluate ensembles of **trained**
machine learning (ML) models. The framework will support a wide spectrum of ensemble techniques. 
Currently FCM supports [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating), [boosting](https://web.stanford.edu/~hastie/Papers/samme.pdf) 
and the chain ensemble described in our paper: _An scalable evolutionary-based approach for accelerating inference with DNN ensembles (2020)_.

The way FCM is designed, makes it very interesting to apply in problems where the the user can generate/has a number
of initial candidate ML solutions to the problem.  The pool can be composed by heterogeneous ML models.  The framework allows the user to test at high rate ensembles  to potentially generate an improved combined solution.
Due to the large number of ensembles that can result from the combination of the initial solution set, FCM adopts a **cheap representation for the ML models**. With the cheap representation we don't need to perform inference
on the actual model, which is costly in DNN. 

Finally, this **repository also contains a genetic algorithm we call EARN**. EARN is built on top of the FCM framework 
so that (guided by a multi-objective cost function),  reduces the number of ensemble combinations to perform while still finding 
close-to-optimal ensembles.

The following sections provide explanations and examples on how to build an ensemble with FCM, how to evaluate it and 
how to load the ML classifiers.


## Design of ensemble methods
An ensemble of ML models is represented as a Direct Acyclic Graph (DAG). The protobuf specification located at *source/FastComposedModels.proto*, describes the available nodes/components of the ensemble/graph along with its members and properties.
The class of the ensemble is found in *source/system_builder*. The python class provides an easy interface for manipulating the protobuf definition of the graph.  
The code that evaluates the DAG and therefore, the ensemble performance on an evaluation set 
can be found in *source/system_evaluator*.


## The Components of an Ensemble

#### The Classifier
A basic component of the ensemble is a classifier. Any kind of ML model that given an input maps it to a label.
As stated previously, we benefit from a cheaper and handier representation of the ML solutions to accelerate the
evaluation and building of ensembles. Beneath there is the representation of a classifier:

- Pre-computed predictions for each input sample of the dataset
    - Train
        - NxC Numpy ndarray. N=number of train samples; C=number of labels
    - Test
        - NxC Numpy ndarray.
    - Validation
        - NxC Numpy ndarray.
- Performance information:
    - Expected inference time 
        - batch_size
        - min/max/avg
    - Number of parameters
    - Number of operations during prediction

The function ``make_classifier()`` in source/make_utils.py, returns a python dictionary with the above structure. 
The user just has to save the dictionary, and reference it later when adding it to an ensemble.

#### The Trigger

The trigger is a component in the ensemble that aims to select the optimal ML model (classifier) for each input. The
trigger can be  ML model or an analytical model. Unlike the classifier, the behaviour of the Trigger is implemented on
a python script file. Please look at the 

The behaviour of a trigger is defined as a python script file that implements the method *train_fit()*. All the information to control the behaviour of the trigger definition, needs to be stored
together in the form of the dataset (Data component in the DAG). Datasets are stored in the folder Data. The method *train_fit()* trains a trigger, and performs inference over the input; This is selecting which classifier to activate.
The method then returns a classifier definition (previous section), saying which classifier needs to be triggered for each input,
and what is the inference time of the trigger, the number of parameters and the number of operations.
 
An example of trigger definition can be seen in *Definitions/Triggers/probability.py*. 

*Tests/test_train_trigger_automatically.py* provides an example of how a trigger is defined in the 
framework, how is added to the graph, how is trained automatically and how to interpret the results 

#### The Merger

Takes the prediction of multiple classifier nodes and regarding the merging protocol, 
provides a single output. Merging techniques available:
    * AVERAGE: Average the logits of the set of networks
    * VOTING: Max voting on the predicted labels of the networks
    * MAX: Select the class that has the maximum global value in the logits
    
#### The Data

Provides meta-information on the dataset the ensemble method is trying to solve. 
    

## Evaluation of an Ensemble
Code for evaluation located in *Source/system_evaluator.py*.
Function evaluate() in the code evaluates the ensemble of NN by receiving as input parameters the graph
describing the ensemble, the set of inputs (ids) to evaluate, and the starting node of the DAG
to perform evaluation on.

#### Evaluation result format
The result of the evaluation is a struct with the following format:  
* test (dict)
    * "system": (struct)
        * accuracy
        * time
        * time_min
        * time_max
        * flops
        * params
        * instances (number of instances executed for the component)
        * instance_model (which model executes what instance)
        * ...
    * "ResNet101": (struct)
        * (same as 'system')
    * "MobileNet-V2": (struct)
        * (same as 'system')
* train (dict)
    * (same content as test)
 
Where 'system' is the performance of the ensemble technique on the input. 
The other elements in the dict of test/train contain the performance of the classifiers/NN
in the ensemble for the **subset** of the input predicted by them.

****Please look at Source/system_evaluator.py for more details**


## Experiments performed on the framework and results
In the folder Example of the project, there can be found several experiments with results saved as plots
of each of them. Each experiment should containing a Readme explaining how to interpret the results already computed, 
and explaining briefly the goals of the experiment.


# Setup
Framework written in **Python3**

To work with this project, **define**:  
1. export PYTHONPATH=$PYTHONPATH:absolute/path/to/FastComposedModels 
2. export FCM=absolute/path/to/FastComposedModels


