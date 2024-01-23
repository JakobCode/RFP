"""
This script contains all functionalities to map a given PyTorch Module to a 
Rlevance Forward Propagation (RFP) Module and back to Basic Representation. 
"""

import torch
import torchvision

from models.modules_relevance_propagation.batchnorm import BatchNorm2d
from models.modules_relevance_propagation.layers import Linear, Conv2d
from models.modules_relevance_propagation.pooling import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from models.modules_relevance_propagation.relu import ReLU
from models.modules_relevance_propagation.dropout import Dropout

import models.modules_relevance_propagation.functional as F

orig_to_cprop = {torch.nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
                 torch.nn.AvgPool2d: AvgPool2d,
                 torch.nn.Linear: Linear,
                 torch.nn.Conv2d: Conv2d,
                 torch.nn.ReLU: ReLU,
                 torch.nn.MaxPool2d: MaxPool2d,
                 torch.nn.BatchNorm2d: BatchNorm2d,
                 torch.nn.Dropout: Dropout}

cprop_to_orig = {AdaptiveAvgPool2d: torch.nn.AdaptiveAvgPool2d,
                 AvgPool2d: torch.nn.AvgPool2d,
                 Linear: torch.nn.Linear,
                 Conv2d: torch.nn.Conv2d,
                 ReLU: torch.nn.ReLU,
                 MaxPool2d: torch.nn.MaxPool2d,
                 BatchNorm2d: torch.nn.BatchNorm2d,
                 Dropout: torch.nn.Dropout}


def _apply_monkey_patch():
    """
    Aplly mokey patches to addopt methods to relevanec forward propagation.
    
    The list of methods can be extended if needed. In this case also adjust the 
    funtion "_remove_mokey_patch".

    Currently, patches are implemented for the following operations. 

    - Interpolation     torch.nn.functional.interpolate 
    - ReLU              torch.nn.functional.relu
                        torch.relu
    - Concatenation     torch.cat
    - Flatten           torch.nn.functional.flatten
                        torch.flatten
    """
    if hasattr(torch.nn.functional, "interpolate_orig"):
        print("Monkey patches already applied.")
    else:

        # Interpolation
        torch.nn.functional.interpolate_orig = torch.nn.functional.interpolate
        torch.nn.functional.interpolate = F.interpolate

        # ReLU
        torch.nn.functional.relu_orig = torch.nn.functional.relu
        torch.relu_orig = torch.relu
        torch.nn.functional.relu = F.relu
        torch.relu = F.relu

        # Concatenation
        torch.cat_orig = torch.cat
        torch.cat = F.cat

        # Flatten
        torch.flatten_orig = torch.flatten
        torch.nn.functional.flatten_orig = torch.flatten
        torch.flatten = F.flatten
        torch.functional.flatten = F.flatten

        print("Applied monkey patches for relevance forward propagation.")

def _remove_mokey_patch():
    """
    Remove mokey patches that addopt methods to relevance forward propagation. 
    This method should invert all patches implemented in "_apply_monkey_patch".

    Currently, patches are implemented for the following operations. 

    - Interpolation     torch.nn.functional.interpolate 
    - ReLU              torch.nn.functional.relu
                        torch.relu
    - Concatenation     torch.cat
    - Flatten           torch.nn.functional.flatten
                        torch.flatten
    """
    if hasattr(torch.nn.functional, "interpolate_orig"):

        # Interpolation
        torch.nn.functional.interpolate = torch.nn.functional.interpolate_orig
        del torch.nn.functional.interpolate_orig
       
        # Concatenation
        torch.cat = torch.cat_orig
        del torch.cat_orig

        # Flatten
        torch.flatten = torch.flatten_orig
        del torch.flatten_orig
        torch.nn.functional.flatten = torch.nn.functional.flatten_orig
        del torch.nn.functional.flatten_orig

        # ReLU
        torch.relu = torch.relu_orig
        del torch.relu_orig
        torch.nn.functional.relu = torch.nn.functional.relu_orig
        del torch.nn.functional.relu_orig
        print("Removed all monkey patches for contribution propagation.")
    else:
        print("not monkey patch to reverse")


def _map_modules(model:torch.nn.Module, map_dict:dict, depth=0, verbose=0):
    """
    Maps (sub-) modules in a given module based on a given mapping dictionary. 
    The method works recursively through the module. 

    Arguments: 

    model       torch.nn.Module             Module to be mapped
    map_dict    dict                        Dictionary mapping module types
                                            [orig_to_cprop or cprop_to_orig]
    depth       int                         Depth of the curretn recursion
    verbose     int                         Level of verbose                      
    """

    children_list = list(model.named_children())

    to_cprop = map_dict == orig_to_cprop

    if depth == 0:
        if not to_cprop:
            _remove_mokey_patch()
        elif to_cprop:
            _apply_monkey_patch()
        else:
            raise Exception()

    for module_id in range(len(children_list)):

        # Already replaced
        if type(children_list[module_id][1]) in map_dict.values():
            if verbose > 1:
                print((depth+1)*" " +"Already Replaced: ", children_list[module_id])
                print("")

        elif type(children_list[module_id][1]) in map_dict:
            
            if verbose > 1:
                print(children_list[module_id])
                print((depth+1)*" "+"Old Module: ", type(children_list[module_id][1]))
            
            if  hasattr(map_dict[type(children_list[module_id][1])], "from_torch"):
                assert to_cprop
                model.__setattr__(children_list[module_id][0], map_dict[type(children_list[module_id][1])].from_torch(children_list[module_id][1]))
            
            elif hasattr(children_list[module_id][1], "to_torch"):
                assert not to_cprop
                model.__setattr__(children_list[module_id][0], children_list[module_id][1].to_torch())
            else:
                raise Exception()
                   
            if verbose > 1:
                print((depth+1)*" "+"New Module: ", type(model.__getattr__(children_list[module_id][0])))
                print("")

        elif hasattr(children_list[module_id][1], "children"):
            if verbose > 0:
                print((depth+1)*" "+"Call Recursive: ", children_list[module_id])
            model.__setattr__(children_list[module_id][0], _map_modules(children_list[module_id][1], map_dict=map_dict, depth=depth+1))
           
    return model

def to_basic_representation(model:torch.nn.Module, verbose=2):
    """
    Map a given Module (from relevance propagation) back to an 
    implementation based on preimplemented Pytorch Modules and 
    functionals.

    Arguments: 
      model           torch.nn.Module           Model to be mapped
      verboes         int                       Level of verboes

    Output:    
      torch.nn.Module           Model with basic forward propagation
    """

    return _map_modules(model=model,
                       map_dict=cprop_to_orig,
                       depth=0, 
                       verbose=verbose)


def to_relevance_representation(model:torch.nn.Module, verbose:int=2):
    """
    Map a given Module from implementation based on preimplemented Pytorch Modules
    to provided relevance forward propagation Modules and functionals.

    Arguments: 
      model           torch.nn.Modules          Model to be mapped
      verboes         int                       Level of verboes

    Output:    
      torch.nn.Module           Model with relevance forward propagation
    """

    return _map_modules(model=model,
                       map_dict=orig_to_cprop,
                       depth=0, 
                       verbose=verbose)


def test_network_mapper():
    """
    This function can be utilized to test the functionality of the network mapper.
    In the current implementation it loads a densenet121 from the torchvision library
    and tests the equality of the results with (1) no changes in the network (2) to_relevance_representation
    applied and (3) to to_basic_representation applied 
    """

    x = torch.rand([3,4,3,128,128], dtype=torch.float64)

    model = torchvision.models.densenet121(pretrained=False).double()
    model = model.eval()
    a = model(x.sum(0))    

    model2 = to_basic_representation(model=model).double()
    model2 = model2.eval()
    b = model2(x.sum(0))
    
    model3 = to_relevance_representation(model=model2).double()
    model3 = model3.eval()
    c = model3(x.sum(0))
    _ = model3(x)
    model4 = to_relevance_representation(model=model3).double()
    model4 = model4.eval()
    d = model4(x)


    model5 = to_basic_representation(model=model4).double()
    model5 = model5.eval()
    e = model5(x.sum(0))

    print("###############################")
    print("    Model 1: Original model")
    print("    Model 2: to_basic_representation applied")
    print()
    print("    Shape output1: ", a.shape)
    print("    Shape output2: ", b.shape)
    print()
    print("    Mean L1-error: ", (a-b).abs().mean().item()) 
    print("    Max L1-error:  ", (a-b).abs().max().item())
    print("###############################")
    print()


    print("###############################")
    print("    Model 1: to_basic_representation applied to original model")
    print("    Model 2: to_relevance_representation applied to Model 1")
    print()
    print("    Shape output1: ", b.shape)
    print("    Shape output2: ", c.shape)
    print()
    print("    Mean L1-error: ", (b-c).abs().mean().item()) 
    print("    Max L1-error:  ", (b-c).abs().max().item())
    print("###############################")
    print()


    print("###############################")
    print("    Model 2: to_relevance_representation applied to Model 1")
    print("    Model 3: to_relevance_representation applied to Model 2")
    print()
    print("    Shape output1: ", c.shape)
    print("    Shape output2: ", d.shape)
    print()
    print("    Mean L1-error: ", (c-d).abs().mean().item()) 
    print("    Max L1-error:  ", (c-d).abs().max().item())
    print("###############################")
    print()


    print("###############################")
    print("    Model 3: to_relevance_representation applied to Model 2")
    print("    Model 4: to_basic_representation applied to Model 3")
    print()
    print("    Shape output1: ", d.shape)
    print("    Shape output2: ", e.shape)
    print()
    print("    Mean L1-error: ", (d-e).abs().mean().item()) 
    print("    Max L1-error:  ", (d-e).abs().max().item())
    print("###############################")
    print()


    print("###############################")
    print("    Model 4: to_basic_representation applied to Model 3")
    print("    Model 1: Original model")
    print()
    print("    Shape output1: ", e.shape)
    print("    Shape output2: ", a.shape)
    print()
    print("    Mean L1-error: ", (e-a).abs().mean().item()) 
    print("    Max L1-error:  ", (e-a).abs().max().item())
    print("###############################")
    print()