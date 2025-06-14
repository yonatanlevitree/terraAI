import React, { useState, useEffect } from 'react';
import { Play, Clock, CheckCircle, XCircle, Loader, Trash2, Zap, Activity, TrendingUp, AlertTriangle, Globe, Link } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE_URL = process.env.REACT_APP_API_URL;
// API functions
const api = {
  createJob: async (jobType, inputData) => {
    const response = await fetch(`${API_BASE_URL}/simulation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputData),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to create job');
    }
    
    const data = await response.json();
    console.log('Created job:', data);
    return data;
  },
  
  listJobs: async (jobType) => {
    const response = await fetch(`${API_BASE_URL}/simulation`);
    if (!response.ok) {
      throw new Error('Failed to fetch jobs');
    }
    const data = await response.json();
    console.log('Listed jobs:', data);
    return data.jobs;
  },
  
  deleteJob: async (jobType, jobId) => {
    const response = await fetch(`${API_BASE_URL}/simulation/${jobId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete job');
    }
    console.log('Deleted job:', jobId);
  }
};

// Validation functions
const validatesimulateInputs = (input1, input2) => {
  const errors = {};
  
  if (!input1 || input1.trim().length === 0) {
    errors.input1 = 'input1 is required and cannot be empty';
  }
  
  if (input2 === '' || input2 === null || input2 === undefined) {
    errors.input2 = 'input2 is required';
  }
  
  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

const validateJob2Inputs = (url, maxRetries, timeout) => {
  const errors = {};
  
  if (!url) {
    errors.url = 'URL is required';
  } else if (!url.match(/^https?:\/\/.+/)) {
    errors.url = 'URL must start with http:// or https://';
  }
  
  if (maxRetries === '' || maxRetries === null || maxRetries === undefined) {
    errors.maxRetries = 'Max retries is required';
  } else if (isNaN(maxRetries) || maxRetries < 0 || maxRetries > 5) {
    errors.maxRetries = 'Max retries must be between 0 and 5';
  }
  
  if (timeout === '' || timeout === null || timeout === undefined) {
    errors.timeout = 'Timeout is required';
  } else if (isNaN(timeout) || timeout < 1 || timeout > 300) {
    errors.timeout = 'Timeout must be between 1 and 300 seconds';
  }
  
  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

const validateSimulateInputs = (terrainSize, maxIterations, depthBounds, volumeBounds, monetaryLimit, timeLimit) => {
  const errors = {};
  if (!terrainSize || terrainSize < 100 || terrainSize > 1000) {
    errors.terrainSize = 'Terrain size must be between 100 and 1000';
  }
  if (maxIterations && (maxIterations < 1 || maxIterations > 1000)) {
    errors.maxIterations = 'Max iterations must be between 1 and 1000';
  }
  if (depthBounds) {
    if (depthBounds[0] >= depthBounds[1] || depthBounds[0] < 0) {
      errors.depthBounds = 'Invalid depth bounds: min must be less than max and non-negative';
    } else if (depthBounds[1] < 10 || depthBounds[1] > 300) {
      errors.depthBounds = 'Max depth must be between 10 and 300 feet';
    }
  }
  if (volumeBounds && (volumeBounds[0] >= volumeBounds[1] || volumeBounds[0] < 0)) {
    errors.volumeBounds = 'Invalid volume bounds: min must be less than max and non-negative';
  }
  if (monetaryLimit && monetaryLimit <= 0) {
    errors.monetaryLimit = 'Monetary limit must be greater than 0';
  }
  if (timeLimit && timeLimit <= 0) {
    errors.timeLimit = 'Time limit must be greater than 0';
  }
  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

const StatusBadge = ({ status }) => {
  const configs = {
    pending: { 
      icon: Clock, 
      color: 'bg-amber-100 text-amber-800 border-amber-200',
      pulse: false
    },
    running: { 
      icon: Loader, 
      color: 'bg-blue-100 text-blue-800 border-blue-200',
      pulse: true
    },
    completed: { 
      icon: CheckCircle, 
      color: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      pulse: false
    },
    failed: { 
      icon: XCircle, 
      color: 'bg-red-100 text-red-800 border-red-200',
      pulse: false
    }
  };
  
  const config = configs[status];
  const IconComponent = config.icon;
  
  return (
    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${config.color}`}>
      <IconComponent className={`w-4 h-4 mr-2 ${config.pulse ? 'animate-spin' : ''}`} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </div>
  );
};

const SimulationProgress = ({ metrics }) => {
  if (!metrics) return null;

  console.log('Received metrics:', metrics);

  // Create x-axis labels from wellsPlaced array
  const xLabels = Array.isArray(metrics.wellsPlaced) ? metrics.wellsPlaced.map((_, index) => `Well ${index + 1}`) : [];

  const chartData = {
    labels: xLabels,
    datasets: [
      {
        label: 'Progress MSE',
        data: Array.isArray(metrics.meanSquaredError) ? metrics.meanSquaredError : [],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        yAxisID: 'y'
      },
      {
        label: 'Monetary Cost ($)',
        data: Array.isArray(metrics.monetaryCost) ? metrics.monetaryCost : [],
        borderColor: 'rgb(54, 162, 235)',
        tension: 0.1,
        yAxisID: 'y1'
      }
    ]
  };

  const options = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Simulation Progress (Note: Final error shown in job card uses weighted MSE)'
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Wells Placed'
        },
        ticks: {
          callback: function(value, index) {
            return Array.isArray(metrics.wellsPlaced) ? metrics.wellsPlaced[index] : '';
          }
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Mean Squared Error'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Monetary Cost ($)'
        },
        grid: {
          drawOnChartArea: false
        }
      }
    }
  };

  return (
    <div className="w-full h-64">
      <Line data={chartData} options={options} />
    </div>
  );
};

const JobCard = ({ job, onDelete }) => {
  const [progressMetrics, setProgressMetrics] = useState(null);
  const [inputData, setInputData] = useState(null);
  const [results, setResults] = useState(null);
  const [currentJob, setCurrentJob] = useState(job);

  useEffect(() => {
    setCurrentJob(job);
  }, [job]);

  useEffect(() => {
    let interval;
    let finished = false;
    const fetchJob = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/simulation/${currentJob.id}`);
        if (response.ok) {
          const updatedJob = await response.json();
          setCurrentJob(updatedJob);
          if (updatedJob.progress) {
            setProgressMetrics(updatedJob.progress);
          }
          if (updatedJob.result) {
            setResults(updatedJob.result);
          }
          if ((updatedJob.status === 'completed' || updatedJob.status === 'failed') && !finished) {
            finished = true;
            clearInterval(interval);
            // Fetch one last time to ensure final state
            setTimeout(fetchJob, 0);
          }
        }
      } catch (err) {
        // Optionally handle error
      }
    };
    if (currentJob && currentJob.status === 'running') {
      fetchJob(); // Fetch immediately
      interval = setInterval(fetchJob, 1000);
    }
    return () => clearInterval(interval);
  }, [currentJob]);

  useEffect(() => {
    if (currentJob) {
      if (currentJob.progress) {
        setProgressMetrics(currentJob.progress);
      }
      if (currentJob.input_data) {
        setInputData(currentJob.input_data);
      }
      if (currentJob.result) {
        setResults(currentJob.result);
      }
    }
  }, [currentJob]);

  const getRuntime = () => {
    if (!job.started_at || !job.completed_at) return 'N/A';
    const start = new Date(job.started_at);
    const end = new Date(job.completed_at);
    const diff = end - start;
    return `${Math.round(diff / 1000)}s`;
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleString();
  };

  if (!job) return null;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-4">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Simulation Job {job.id}
          </h3>
          {job.parameters && job.parameters.algorithm && (
            <span className="inline-block mt-1 px-2 py-1 text-xs font-semibold rounded bg-blue-100 text-blue-800">
              Algorithm: {job.parameters.algorithm.charAt(0).toUpperCase() + job.parameters.algorithm.slice(1)}
            </span>
          )}
          <p className="text-sm text-gray-500">
            Created: {job.created_at ? formatDate(job.created_at) : 'N/A'}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            job.status === 'completed' ? 'bg-green-100 text-green-800' :
            job.status === 'running' ? 'bg-blue-100 text-blue-800' :
            job.status === 'failed' ? 'bg-red-100 text-red-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {job.status}
          </span>
          <button
            onClick={() => onDelete(job.id)}
            className="p-2 text-gray-400 hover:text-red-500 transition-colors"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      {job.status === 'running' && (
        <div className="mb-4">
          <div className="flex items-center text-sm text-gray-500">
            <Clock className="w-4 h-4 mr-1" />
            Running for {getRuntime()}
          </div>
        </div>
      )}

      {/* Always show input parameters for the job, regardless of status */}
      {job.parameters && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Input Parameters</h4>
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
            {Object.entries(job.parameters).map(([key, value]) => (
              <div key={key} className="flex">
                <span className="font-semibold mr-2">{key}:</span>
                <span>{Array.isArray(value) ? `[${value.join(', ')}]` : String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {inputData && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Input Data</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Terrain Size: {inputData.terrainSize}</p>

            </div>
            <div>
              <p className="text-sm text-gray-600">Max Iterations: {inputData.maxIterations}</p>
              <p className="text-sm text-gray-600">Depth Bounds: [{inputData.depthBounds[0]}, {inputData.depthBounds[1]}]</p>
              <p className="text-sm text-gray-600">Volume Bounds: [{inputData.volumeBounds[0]}, {inputData.volumeBounds[1]}]</p>
            </div>
          </div>
        </div>
      )}

      {progressMetrics && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Progress</h4>
          {job.status === 'completed' && Array.isArray(progressMetrics.meanSquaredError) && progressMetrics.meanSquaredError.length > 0 && (
            <div className="mb-2">
              <p className="text-sm text-gray-600">
                Final Error: {progressMetrics.meanSquaredError[progressMetrics.meanSquaredError.length - 1].toFixed(4)}
              </p>
            </div>
          )}
          <SimulationProgress metrics={progressMetrics} />
        </div>
      )}

      {results && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">Results</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Wells Placed: {results.terrain_summary && typeof results.terrain_summary.totalWells === 'number' ? results.terrain_summary.totalWells : 'N/A'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Cost: {results.terrain_summary && typeof results.terrain_summary.monetaryCost === 'number' ? `$${results.terrain_summary.monetaryCost.toFixed(2)}` : 'N/A'}</p>
              <p className="text-sm text-gray-600">Total Time: {results.terrain_summary && typeof results.terrain_summary.timeCost === 'number' ? `${results.terrain_summary.timeCost.toFixed(2)} units` : 'N/A'}</p>
            </div>
          </div>
        </div>
      )}

      {job.error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="text-sm font-medium text-red-700 mb-2">Error</h4>
          <p className="text-sm text-red-800">{job.error}</p>
        </div>
      )}

      {/* Always show genetic parameters for genetic/geneticSingle jobs, with N/A fallback */}
      {job.parameters && (job.parameters.algorithm === 'genetic' || job.parameters.algorithm === 'geneticSingle') && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-medium text-blue-700 mb-2">Genetic Algorithm Parameters</h4>
          <div className="grid grid-cols-2 gap-2 text-sm text-blue-900">
            <div><span className="font-semibold mr-2">Num Generations:</span>{job.parameters.numGenerations ?? 'N/A'}</div>
            <div><span className="font-semibold mr-2">Mutation Rate:</span>{job.parameters.mutationRate ?? 'N/A'}</div>
            <div><span className="font-semibold mr-2">Tournament Size:</span>{job.parameters.tournamentSize ?? 'N/A'}</div>
            <div><span className="font-semibold mr-2">Elite Size:</span>{job.parameters.eliteSize ?? 'N/A'}</div>
            <div><span className="font-semibold mr-2">Population Size:</span>{job.parameters.populationSize ?? 'N/A'}</div>
          </div>
        </div>
      )}
    </div>
  );
};

const StatCard = ({ icon: Icon, label, value, color, trend }) => (
  <div className={`bg-white rounded-xl p-6 shadow-sm border border-gray-100 ${color}`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600">{label}</p>
        <p className="text-3xl font-bold text-gray-900 mt-1">{value}</p>
      </div>
      <div className={`p-3 rounded-full ${color.replace('border-', 'bg-').replace('-200', '-100')}`}>
        <Icon className={`w-6 h-6 ${color.replace('border-', 'text-').replace('-200', '-600')}`} />
      </div>
    </div>
    {trend && (
      <div className="flex items-center mt-3 text-sm">
        <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
        <span className="text-green-600">{trend}</span>
      </div>
    )}
  </div>
);

export default function AsyncJobManager() {
  const [selectedJobType, setSelectedJobType] = useState('simulate');
  
  // Terrain simulation inputs
  const [terrainSize, setTerrainSize] = useState('400');
  const [noise, setnoise] = useState('0.5');
  const [smoothness, setSmoothness] = useState('0.5');
  const [maxIterations, setMaxIterations] = useState('50');
  const [depthMin, setDepthMin] = useState('5');
  const [depthMax, setDepthMax] = useState('300');
  const [volumeMin, setVolumeMin] = useState('10');
  const [volumeMax, setVolumeMax] = useState('500');
  const [monetaryLimit, setMonetaryLimit] = useState('5000000');
  const [timeLimit, setTimeLimit] = useState('500000');
  const [fidelity, setFidelity] = useState('0.9');
  const [algorithm, setAlgorithm] = useState('greedy');
  const [seed, setSeed] = useState('');
  
  // Genetic/GeneticSingle-specific fields
  const [numGenerations, setNumGenerations] = useState('100');
  const [mutationRate, setMutationRate] = useState('0.1');
  const [tournamentSize, setTournamentSize] = useState('6');
  const [eliteSize, setEliteSize] = useState('3');
  const [populationSize, setPopulationSize] = useState('50');
  
  const [jobs, setJobs] = useState([]);
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const loadJobs = async () => {
    try {
      const simulateList = await api.listJobs('simulate');
      setJobs(simulateList.sort((a, b) => 
        new Date(b.created_at) - new Date(a.created_at)
      ));
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to load jobs:', error);
    }
  };

  useEffect(() => {
    loadJobs();
    
    // Auto-refresh every 2 seconds
    const interval = setInterval(loadJobs, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async () => {
    try {
      // Parse all inputs to numbers
      const parsedInputs = {
        fidelity: parseFloat(fidelity),
        terrainSize: parseInt(terrainSize),
        maxIterations: parseInt(maxIterations),
        depthBounds: [parseFloat(depthMin), parseFloat(depthMax)],
        volumeBounds: [parseFloat(volumeMin), parseFloat(volumeMax)],
        monetaryLimit: parseFloat(monetaryLimit),
        timeLimit: parseFloat(timeLimit),
        algorithm: algorithm,
        seed: seed === '' ? undefined : parseInt(seed)
      };
      // Add genetic/geneticSingle fields if relevant
      if (algorithm === 'genetic' || algorithm === 'geneticSingle') {
        parsedInputs.numGenerations = parseInt(numGenerations);
        parsedInputs.mutationRate = parseFloat(mutationRate);
        parsedInputs.tournamentSize = parseInt(tournamentSize);
        parsedInputs.eliteSize = parseInt(eliteSize);
        parsedInputs.populationSize = parseInt(populationSize);
      }
      // Validate all inputs
      const validation = validateSimulateInputs(
        parsedInputs.terrainSize,
        parsedInputs.maxIterations,
        parsedInputs.depthBounds,
        parsedInputs.volumeBounds,
        parsedInputs.monetaryLimit,
        parsedInputs.timeLimit
      );
      if (!validation.isValid) {
        setErrors(validation.errors);
        return;
      }
      setIsSubmitting(true);
      setErrors({});
      const inputData = {
        fidelity: parsedInputs.fidelity,
        terrainSize: parsedInputs.terrainSize,
        maxIterations: parsedInputs.maxIterations,
        depthBounds: parsedInputs.depthBounds,
        volumeBounds: parsedInputs.volumeBounds,
        monetaryLimit: parsedInputs.monetaryLimit,
        timeLimit: parsedInputs.timeLimit,
        algorithm: parsedInputs.algorithm,
        seed: parsedInputs.seed
      };
      console.log('Submitting job with data:', inputData);

      // Add genetic/geneticSingle fields if relevant
      if (algorithm === 'genetic' || algorithm === 'geneticSingle') {
        inputData.numGenerations = parsedInputs.numGenerations;
        inputData.mutationRate = parsedInputs.mutationRate;
        inputData.tournamentSize = parsedInputs.tournamentSize;
        inputData.eliteSize = parsedInputs.eliteSize;
        inputData.populationSize = parsedInputs.populationSize;
      }
      await api.createJob('simulate', inputData);
      // Clear form
      setFidelity('0.9');
      setTerrainSize('400');
      setMaxIterations('100');
      setDepthMin('5');
      setDepthMax('300');
      setVolumeMin('10');
      setVolumeMax('5000');
      setMonetaryLimit('50000000');
      setTimeLimit('500000');
      setAlgorithm('greedy');
      setSeed('rw');
      // Refresh job list
      await loadJobs();
    } catch (error) {
      console.error('Error creating job:', error);
      setErrors({ submit: error.message || 'Failed to create job. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (jobId) => {
    try {
      await api.deleteJob('simulate', jobId);
      await loadJobs();
    } catch (error) {
      console.error('Failed to delete job:', error);
    }
  };

  const stats = {
    total: jobs.length,
    running: jobs.filter(job => job.status === 'running').length,
    completed: jobs.filter(job => job.status === 'completed').length,
    failed: jobs.filter(job => job.status === 'failed').length
  };

  const successRate = stats.total > 0 ? Math.round((stats.completed / (stats.completed + stats.failed)) * 100) || 0 : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Terrain Simulation Manager</h1>
                <p className="text-sm text-gray-500">Async terrain simulation dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatCard 
            icon={Activity} 
            label="Total Jobs" 
            value={stats.total} 
            color="border-blue-200"
          />
          <StatCard 
            icon={Loader} 
            label="Running" 
            value={stats.running} 
            color="border-amber-200"
          />
          <StatCard 
            icon={CheckCircle} 
            label="Completed" 
            value={stats.completed} 
            color="border-emerald-200"
          />
          <StatCard 
            icon={AlertTriangle} 
            label="Success Rate" 
            value={`${successRate}%`} 
            color="border-purple-200"
            trend={successRate >= 80 ? "Good performance" : "Needs attention"}
          />
        </div>
        
        {/* Create Job Form */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8 mb-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center">
                <Play className="w-4 h-4 text-white" />
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Create New Simulation</h2>
            </div>
          </div>
          
          {errors.submit && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
                <p className="text-red-700">{errors.submit}</p>
              </div>
            </div>
          )}
          
          <form onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }} className="space-y-6">
            {/* Algorithm selection at the top, full row */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Algorithm <span className="text-red-500">*</span>
              </label>
              <select
                value={algorithm}
                onChange={e => setAlgorithm(e.target.value)}
                className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                required
              >
                <option value="genetic">Genetic (Full Solution)</option>
                <option value="geneticSingle">Genetic (Single Well)</option>
                <option value="greedy">Greedy</option>
              </select>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Terrain Size <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="100"
                  max="1000"
                  value={terrainSize}
                  onChange={(e) => {
                    setTerrainSize(e.target.value);
                    if (errors.terrainSize) {
                      setErrors(prev => ({ ...prev, terrainSize: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.terrainSize ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                  placeholder="100-1000"
                />
                {errors.terrainSize && (
                  <p className="text-red-500 text-sm mt-2 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {errors.terrainSize}
                  </p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Iterations <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={maxIterations}
                  onChange={(e) => {
                    setMaxIterations(e.target.value);
                    if (errors.maxIterations) {
                      setErrors(prev => ({ ...prev, maxIterations: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.maxIterations ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
                {errors.maxIterations && (
                  <p className="text-red-500 text-sm mt-2 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {errors.maxIterations}
                  </p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Depth Min <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={depthMin}
                  onChange={(e) => {
                    setDepthMin(e.target.value);
                    if (errors.depthBounds) {
                      setErrors(prev => ({ ...prev, depthBounds: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.depthBounds ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
                {errors.depthBounds && (
                  <p className="text-red-500 text-sm mt-2 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {errors.depthBounds}
                  </p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Depth Max <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0"
                  max="500"
                  value={depthMax}
                  onChange={(e) => {
                    setDepthMax(e.target.value);
                    if (errors.depthBounds) {
                      setErrors(prev => ({ ...prev, depthBounds: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.depthBounds ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Volume Min <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0"
                  max="1000"
                  value={volumeMin}
                  onChange={(e) => {
                    setVolumeMin(e.target.value);
                    if (errors.volumeBounds) {
                      setErrors(prev => ({ ...prev, volumeBounds: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.volumeBounds ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
                {errors.volumeBounds && (
                  <p className="text-red-500 text-sm mt-2 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {errors.volumeBounds}
                  </p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Volume Max <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0"
                  max="1000"
                  value={volumeMax}
                  onChange={(e) => {
                    setVolumeMax(e.target.value);
                    if (errors.volumeBounds) {
                      setErrors(prev => ({ ...prev, volumeBounds: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.volumeBounds ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Monetary Limit <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0"
                  value={monetaryLimit}
                  onChange={(e) => {
                    setMonetaryLimit(e.target.value);
                    if (errors.monetaryLimit) {
                      setErrors(prev => ({ ...prev, monetaryLimit: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.monetaryLimit ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
                {errors.monetaryLimit && (
                  <p className="text-red-500 text-sm mt-2 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {errors.monetaryLimit}
                  </p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Time Limit <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0"
                  value={timeLimit}
                  onChange={(e) => {
                    setTimeLimit(e.target.value);
                    if (errors.timeLimit) {
                      setErrors(prev => ({ ...prev, timeLimit: null }));
                    }
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.timeLimit ? 'border-red-300 bg-red-50' : 'border-gray-200'
                  }`}
                />
                {errors.timeLimit && (
                  <p className="text-red-500 text-sm mt-2 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {errors.timeLimit}
                  </p>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Fidelity <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  min="0.1"
                  max="1.0"
                  step="0.01"
                  value={fidelity}
                  onChange={(e) => {
                    setFidelity(e.target.value);
                  }}
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors`}
                  placeholder="0.1-1.0"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Seed (optional)
                </label>
                <input
                  type="number"
                  value={seed}
                  onChange={e => setSeed(e.target.value)}
                  className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  placeholder="Random if left blank"
                />
              </div>
            </div>
            
            {/* Genetic/GeneticSingle-specific fields */}
            {(algorithm === 'genetic' || algorithm === 'geneticSingle') && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Generations
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={numGenerations}
                    onChange={e => setNumGenerations(e.target.value)}
                    className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="e.g. 100"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Mutation Rate
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={mutationRate}
                    onChange={e => setMutationRate(e.target.value)}
                    className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="e.g. 0.1"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Tournament Size
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={tournamentSize}
                    onChange={e => setTournamentSize(e.target.value)}
                    className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="e.g. 6"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Elite Size
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={eliteSize}
                    onChange={e => setEliteSize(e.target.value)}
                    className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="e.g. 3"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Population Size
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={populationSize}
                    onChange={e => setPopulationSize(e.target.value)}
                    className="w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="e.g. 50"
                    required
                  />
                </div>
              </div>
            )}
            
            <div className="mt-6">
              <button
                type="submit"
                disabled={isSubmitting}
                className={`w-full px-4 py-3 rounded-xl transition-colors ${
                  isSubmitting
                    ? 'bg-gray-300 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {isSubmitting ? 'Submitting...' : 'Create Simulation'}
              </button>
            </div>
          </form>
        </div>
        
        {/* Job List */}
        <div className="mt-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Job List</h2>
          {jobs.map(job => (
            <JobCard key={job.id} job={job} onDelete={handleDelete} />
          ))}
        </div>
      </div>
    </div>
  );
}