
import torch
from typing import Tuple, Dict
from pymlrf.types import CriterionProtocol
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from pymlrf.types import (
    GenericDataLoaderProtocol
    )

# STUDENT CODE: Import metric utils
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score

class ValidateSingleEpoch:
    
    def __init__(
        self, 
        half_precision:bool=False,
        cache_preds:bool=True
        ) -> None:
        """Class which runs a single epoch of validation.

        Args:
            half_precision (bool, optional): Boolean defining whether to run in 
            half-precision. Defaults to False.
            cache_preds (bool, optional): Boolean defining whether to save the 
            prediction outputs. Defaults to True.
        """
        self.half_precision = half_precision
        self.cache_preds = cache_preds

    def __call__(
        self,
        model:torch.nn.Module,
        data_loader:GenericDataLoaderProtocol,
        gpu:bool,
        criterion:CriterionProtocol
        )->Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        """ Call function which runs a single epoch of validation
        Args:
            model (BaseModel): Torch model of type BaseModel i.e., it should
            subclass the BaseModel class
            data_loader (DataLoader): Torch data loader object
            gpu (bool): Boolean defining whether to use a GPU if available
            criterion (CriterionProtocol): Criterian to use for training or 
            for training and validation if val_criterion is not specified. I.e., 
            this could be nn.MSELoss() 
            
        Returns:
            Tuple[torch.Tensor, Dict[str,torch.Tensor]]: Tuple defining the 
            final loss for the epoch and a dictionary of predictions. The keys 
            will be the same keys required by the criterion. 
        """

        losses = torch.tensor(0.0)
        denom = torch.tensor(0)
        if gpu:
            _device = "cuda"
        else:
            _device = "cpu"
            
        if self.half_precision:
            losses = losses.half()
            denom = denom.half()
        model.eval()
        preds = []

        # STUDENT CODE: Define array for actual values
        actuals = []

        with torch.no_grad():
            for i, vals in enumerate(data_loader):

                # Prepare input data
                input_vals = vals.input
                output_vals = vals.output
                if gpu:
                    input_vals = {
                        val:input_vals[val].cuda() for val in input_vals
                        }
                    output_vals = {
                        val:output_vals[val].cuda() for val in output_vals
                        }
                else:
                    input_vals = {
                        val:Variable(input_vals[val]) for val in input_vals
                        }
                    output_vals = {
                        val:Variable(output_vals[val]) for val in output_vals
                        }

                # Compute output
                if self.half_precision:
                    with torch.autocast(device_type=_device):
                        output = model(**input_vals)
                else:
                    output = model(**input_vals)

                # Logs
                val_loss = criterion(output, output_vals)
                losses += val_loss.detach().cpu()
                denom += 1
                if self.cache_preds:
                    # STUDENT CODE: Commented out as it is ran anyway for metric collection
                    pass
                    # preds.append({k:output[k].detach().cpu() for k in output.keys()})

                # STUDENT CODE: Keep track of prediction and actual values for each prediction
                preds.append({k:output[k].detach().cpu() for k in output.keys()})
                actuals.append({k: output_vals[k].detach().cpu() for k in output_vals.keys()})


        _prd_lst = {}

        # STUDENT CODE: Define dictionary for truth values for comparison
        _act_lst = {}

        # STUDENT CODE: Append prediction and actual values to dictionary
        for k in preds[0].keys():
            _prd_lst[k] = torch.concat([t[k] for t in preds],dim=0)
            _act_lst[k] = torch.concat([t[k] for t in actuals], dim=0)

        if self.cache_preds:
            # STUDENT CODE: Commented out as it is ran anyway for metric collection
            pass
            # for k in preds[0].keys():
            #     _prd_lst[k] = torch.concat([t[k] for t in preds],dim=0)

        losses = losses/denom

        # STUDENT CODE: Convert classification predictions for metrics
        if "grp" in _prd_lst:
            if len(_prd_lst["grp"].shape) > 1: 
                _prd_lst["grp"] = torch.argmax(_prd_lst["grp"], dim=1)
            if len(_act_lst["grp"].shape) > 1:
                _act_lst["grp"] = torch.argmax(_act_lst["grp"], dim=1)

        # STUDENT CODE: Compute metrics
        metrics = {}
        metrics["r2_pos"] = r2_score(_act_lst["pos"].numpy(), _prd_lst["pos"].numpy())
        metrics["mse_pos"] = mean_squared_error(_act_lst["pos"].numpy(), _prd_lst["pos"].numpy())
        metrics["accuracy_grp"] = accuracy_score(_act_lst["grp"].numpy(), _prd_lst["grp"].numpy())
        metrics["precision_grp"] = precision_score(
            _act_lst["grp"].numpy(), 
            _prd_lst["grp"].numpy(), 
            average="weighted"
        )
        metrics["recall_grp"] = recall_score(
            _act_lst["grp"].numpy(), 
            _prd_lst["grp"].numpy(), 
            average="weighted"
        )

        return losses, _prd_lst, metrics