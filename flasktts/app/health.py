import torch
from flask_restx import Namespace, Resource

api = Namespace("health", description="Health check endpoints")


@api.route("/check")
class HealthCheck(Resource):
    @api.doc(responses={200: "API is healthy"})
    def get(self):
        """Check if the API is healthy"""
        return {"status": "healthy", "version": "0.0.1"}


@api.route("/gpu")
class GPUCheck(Resource):
    @api.doc(responses={200: "GPU is available"})
    def get(self):
        """Check if a GPU is available"""
        cuda = torch.cuda.is_available()
        cudnn = torch.backends.cudnn.enabled
        return {
            "cuda": cuda,
            "cudnn": cudnn,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else "",
            "n_gpus": torch.cuda.device_count() if cuda else 0,
            "mps": torch.backends.mps.is_available(),
        }
