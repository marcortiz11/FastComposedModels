[![Generic badge](https://img.shields.io/github/license/marcortiz11/FastComposedModels)](https://shields.io/)
[![Generic badge](https://img.shields.io/github/v/release/marcortiz11/FastComposedModels)](https://shields.io/)

# FastComposedModels
Framework that allows to build and evaluate efficiently ensemble methods for neural networks by considering the evaluation results
of the network (logits, inference time, params, flops) in train and test in classification tasks.

## Design of ensemble methods
The ensemble of NNs in the framework translates into the definition, building and evaluation of a DAG; 
The definition of the graph with its available components/nodes and constraints
are specified in a .proto definition found in *Source/FastComposedModels.proto*.  
The code that builds the DAG can be found in *Source/system_builder*.  
The code that builds the components of the graph is located in *Source/make_util*.
The code that evaluates the DAG and therefore, the ensemble performance on an evaluation set 
can be found in *Source/system_evaluator*.
  
The available components of the graph are:
*  **Classifier**: Classifier nodes receive a set of input ids. Input ids reference inputs on the evaluation set.
The node then returns the predictions (logits, label) for the inputs. Nodes point Classifiers to predict a subset of the test/evaluation set. 
*  **Merger**: Takes the prediction of multiple classifier nodes and regarding the merging protocol, 
provides a single output. Merging techniques available:
    * AVERAGE: Average the logits of the set of networks
    * VOTING: Max voting on the predicted labels of the networks
    * MAX: Select the class that has the maximum global value in the logits
*  **Trigger**: Selects the appropriate network/s that will perform prediction for the current input. 
If trigger is not trained, the evaluation
part of the framework will train the trigger according to the trigger definition.
*  **Data**: Provides meta-information on the dataset the ensemble method is trying to solve. 


## Evaluation 
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

## Defining Classifiers
In order to perform a fast evaluation on the ensemble methods, the framework considers the performance
of the classifiers over a dataset, along with the inference time, #param and #ops measurements of the classifier. 
The classifier is defined as a dictionary with the following information:

- Matrix of logits NxC (N=samples in the evaluation set, C=Number of classes)
- Vector of ground truths N (N=samples in the evaluation set)
- Vector of ids N (N=samples in the evaluation set), linking entries of the logit matrix to the ground truth and
identifying the samples. Vector of ids is crucial, some classifiers may be trained with a subset of the dataset, therefore it is important to identify
which are those inputs.
- Time measurements: List of inference times. Each time measurement corresponds to performing inference with
128 input instances.
- Number of params
- FLOPS

The previous information is saved as a dict and stored as a pickle in *Definitions/Classifiers*. 
A classifier node in the DAG, needs to point to the pickle file for the evaluation part. To build the
dict look at the code *Source/make_utils.py*.


## Defining Triggers

The behaviour of a trigger is defined as a python script file that implements the method *train_fit()*. All the information to control the behaviour of the trigger definition, needs to be stored
together in the form of the dataset (Data component in the DAG). Datasets are stored in the folder Data. The method *train_fit()* trains a trigger, and performs inference over the input; This is selecting which classifier to activate.
The method then returns a classifier definition (previous section), saying which classifier needs to be triggered for each input,
and what is the inference time of the trigger, the number of parameters and the number of operations.
 
An example of trigger definition can be seen in *Definitions/Triggers/probability.py*. 

### Example defining triggers
*Tests/test_train_trigger_automatically.py* provides an example of how a trigger is defined in the 
framework, how is added to the graph, how is trained automatically and how to interpret the results 

## Simple examples for using the framework
In *SimpleSamples/** you can find simple examples


## Experiments performed on the framework and results
In the folder Example of the project, there can be found several experiments with results saved as plots
of each of them. Each experiment should containing a Readme explaining how to interpret the results already computed, 
and explaining briefly the goals of the experiment.


# Setup
Framework written in **Python3**

To work with this project, **define**:  
1. export PYTHONPATH=$PYTHONPATH:absolute/path/to/FastComposedModels 
2. export FCM=absolute/path/to/FastComposedModels

Additionally, work with this framework in the **CLUSTER**:  

Consider creating symlinks instead of saving the classifier definitions.
A python script copy_models.py in the folder *Definitions* of the framework, already
is able to recursively create symlinks to classifiers definitions saved either in 
dataP or dataT (preferred).
 


