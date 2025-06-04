import threading
from functions.v1.optimizers.genetic import GeneticOptimizer
from functions.v1.optimizers.greedy import GreedyOptimizer
from flask import Blueprint, request, jsonify
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from marshmallow import Schema, fields, validate

simulate_bp = Blueprint('simulate', __name__)

# Define the request schema
class SimulationParameters(Schema):
    fidelity = fields.Float(required=True, validate=validate.Range(min=0.1, max=1.0))
    terrainSize = fields.Integer(required=True, validate=validate.Range(min=100, max=1000))
    depthBounds = fields.List(fields.Float(), required=True, validate=validate.Length(equal=2))
    volumeBounds = fields.List(fields.Float(), required=True, validate=validate.Length(equal=2))
    noise = fields.Float(required=True, validate=validate.Range(min=0.0, max=1.0))
    smoothness = fields.Float(required=True, validate=validate.Range(min=0.0, max=1.0))
    name = fields.String(required=False)
    description = fields.String(required=False)
    maxIterations = fields.Integer(required=True, validate=validate.Range(min=1, max=1000))
    monetaryLimit = fields.Integer(required=True, validate=validate.Range(min=1, max=1000000000))
    timeLimit = fields.Integer(required=True, validate=validate.Range(min=1, max=1000000))
    algorithm = fields.String(required=True, validate=validate.OneOf(["genetic", "greedy"]))

# Define the response schema
class SimulationResponse(Schema):
    job_id = fields.String(required=True)
    status = fields.String(required=True)
    parameters = fields.Dict(required=True)

# In-memory storage for jobs (replace with database later)
jobs = {}

@simulate_bp.route('/simulation', methods=['POST'])
def create_simulation():
    """
    Create a new simulation job
    ---
    post:
      summary: Create a new simulation job
      description: Creates a new simulation job with the specified parameters
      requestBody:
        required: true
        content:
          application/json:
            schema: SimulationParameters
      responses:
        200:
          description: Simulation job created successfully
          content:
            application/json:
              schema: SimulationResponse
        400:
          description: Invalid parameters
    """
    try:
        data = request.get_json()
        schema = SimulationParameters()
        validated_data = schema.load(data)
        
        # Generate a simple job ID (replace with UUID in production)
        job_id = f"job_{len(jobs) + 1}"
        
        # Store the job
        jobs[job_id] = {
            "id": job_id,
            "parameters": validated_data,
            "status": "pending"
        }
        
        thread = threading.Thread(target=run_simulation, args=(job_id, validated_data))
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "pending",
            "parameters": validated_data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def run_simulation(job_id, parameters):
    """
    Run a simulation job
    """
    try:
        # Initialize the optimizer
        if parameters['algorithm'] == 'genetic':
            optimizer = GeneticOptimizer(**parameters)
        else:
            optimizer = GreedyOptimizer(**parameters)
        
        # Run the optimization
        result = optimizer.optimize()
        
        # Update the job status
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = result
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


@simulate_bp.route('/simulation/<job_id>', methods=['GET'])
def get_simulation_status(job_id):
    """
    Get simulation job status
    ---
    get:
      summary: Get simulation job status
      description: Retrieve the status of a simulation job
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Job status retrieved successfully
          content:
            application/json:
              schema: SimulationResponse
        404:
          description: Job not found
    """
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
        
    return jsonify(jobs[job_id]), 200

@simulate_bp.route('/simulation', methods=['GET'])
def list_simulation_jobs():
    """
    List all simulation jobs
    ---
    get:
      summary: List all simulation jobs
      description: Returns a list of all simulation jobs
      responses:
        200:
          description: A list of jobs
          content:
            application/json:
              schema:
                type: object
                properties:
                  jobs:
                    type: array
                    items: 
                      $ref: '#/components/schemas/SimulationResponse'
    """
    return jsonify({"jobs": list(jobs.values())}), 200

@simulate_bp.route('/simulation/<job_id>', methods=['DELETE'])
def delete_simulation_job(job_id):
    """
    Delete a simulation job
    ---
    delete:
      summary: Delete a simulation job
      description: Deletes a simulation job by job_id
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Job deleted successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
        404:
          description: Job not found
    """
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    del jobs[job_id]
    return jsonify({"message": f"Job {job_id} deleted successfully."}), 200

def register_routes(app):
    """Register OpenAPI documentation"""
    spec = APISpec(
        title="TerraAI Simulation API",
        version="1.0.0",
        openapi_version="3.0.2",
        plugins=[MarshmallowPlugin()],
    )
    
    # Register schemas
    spec.components.schema("SimulationParameters", schema=SimulationParameters)
    spec.components.schema("SimulationResponse", schema=SimulationResponse)
    
    # Register routes
    with app.test_request_context():
        spec.path(view=create_simulation)
        spec.path(view=get_simulation_status)
    
    return spec 