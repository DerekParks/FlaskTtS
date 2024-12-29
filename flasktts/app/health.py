from flask_restx import Namespace, Resource

api = Namespace('health', description='Health check endpoints')


@api.route('/check')
class HealthCheck(Resource):
    @api.doc(responses={200: 'API is healthy'})
    def get(self):
        """Check if the API is healthy"""
        return {
            'status': 'healthy',
            'version': '0.0.1'
        }
