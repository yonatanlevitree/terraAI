import React, { useState } from 'react';
import axios from 'axios';

interface SimulationParameters {
    terrainSize: number;
    noise: number;
    smoothness: number;
    maxIterations: number;
    depthBounds: [number, number];
    volumeBounds: [number, number];
    monetaryLimit: number;
    timeLimit: number;
    fidelity: number;
    name?: string;
    description?: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL;

const SimulationForm: React.FC = () => {
    const [parameters, setParameters] = useState<SimulationParameters>({
        terrainSize: 500,
        noise: 0.5,
        smoothness: 0.5,
        maxIterations: 100,
        depthBounds: [0, 100],
        volumeBounds: [0, 1000],
        monetaryLimit: 1000000,
        timeLimit: 100000,
        fidelity: 0.5,
        name: '',
        description: ''
    });

    const [jobId, setJobId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [algorithm, setAlgorithm] = useState<string>('greedy');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        
        try {
            const response = await axios.post(`${API_BASE_URL}/simulation`, {
                ...parameters,
                algorithm,
            });
            setJobId(response.data.job_id);
        } catch (err) {
            setError('Failed to create simulation job');
            console.error(err);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setParameters(prev => ({
            ...prev,
            [name]: name.includes('bounds') ? 
                name === 'depthBounds' ? 
                    [parseFloat(value), prev.depthBounds[1]] :
                    [parseFloat(value), prev.volumeBounds[1]] :
                name === 'terrainSize' ? 
                    parseInt(value) : 
                    parseFloat(value)
        }));
    };

    return (
        <div className="max-w-2xl mx-auto p-4">
            <h2 className="text-2xl font-bold mb-4">Create Simulation</h2>
            
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                    {error}
                </div>
            )}
            
            {jobId && (
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
                    Simulation job created! Job ID: {jobId}
                </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block mb-1">Fidelity (0.1-1.0)</label>
                    <input
                        type="number"
                        name="fidelity"
                        value={parameters.fidelity}
                        onChange={handleChange}
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        className="w-full p-2 border rounded"
                        required
                    />
                </div>

                <div>
                    <label className="block mb-1">Terrain Size (100-1000)</label>
                    <input
                        type="number"
                        name="terrainSize"
                        value={parameters.terrainSize}
                        onChange={handleChange}
                        min="100"
                        max="1000"
                        className="w-full p-2 border rounded"
                        required
                    />
                </div>

                <div>
                    <label className="block mb-1">Depth Bounds</label>
                    <div className="flex space-x-2">
                        <input
                            type="number"
                            name="depthBounds"
                            value={parameters.depthBounds[0]}
                            onChange={handleChange}
                            className="w-full p-2 border rounded"
                            required
                        />
                        <input
                            type="number"
                            name="depthBounds"
                            value={parameters.depthBounds[1]}
                            onChange={handleChange}
                            className="w-full p-2 border rounded"
                            required
                        />
                    </div>
                </div>

                <div>
                    <label className="block mb-1">Volume Bounds</label>
                    <div className="flex space-x-2">
                        <input
                            type="number"
                            name="volumeBounds"
                            value={parameters.volumeBounds[0]}
                            onChange={handleChange}
                            className="w-full p-2 border rounded"
                            required
                        />
                        <input
                            type="number"
                            name="volumeBounds"
                            value={parameters.volumeBounds[1]}
                            onChange={handleChange}
                            className="w-full p-2 border rounded"
                            required
                        />
                    </div>
                </div>

                <div>
                    <label className="block mb-1">Noise (0-1)</label>
                    <input
                        type="number"
                        name="noise"
                        value={parameters.noise}
                        onChange={handleChange}
                        min="0"
                        max="1"
                        step="0.1"
                        className="w-full p-2 border rounded"
                        required
                    />
                </div>

                <div>
                    <label className="block mb-1">Smoothness (0-1)</label>
                    <input
                        type="number"
                        name="smoothness"
                        value={parameters.smoothness}
                        onChange={handleChange}
                        min="0"
                        max="1"
                        step="0.1"
                        className="w-full p-2 border rounded"
                        required
                    />
                </div>

                <div>
                    <label className="block mb-1">Name (optional)</label>
                    <input
                        type="text"
                        name="name"
                        value={parameters.name}
                        onChange={handleChange}
                        className="w-full p-2 border rounded"
                    />
                </div>

                <div>
                    <label className="block mb-1">Description (optional)</label>
                    <input
                        type="text"
                        name="description"
                        value={parameters.description}
                        onChange={handleChange}
                        className="w-full p-2 border rounded"
                    />
                </div>

                <div>
                    <label className="block mb-1">Algorithm</label>
                    <select
                        name="algorithm"
                        value={algorithm}
                        onChange={e => setAlgorithm(e.target.value)}
                        className="w-full p-2 border rounded"
                        required
                    >
                        <option value="genetic">Genetic</option>
                        <option value="greedy">Greedy</option>
                    </select>
                </div>

                <button
                    type="submit"
                    className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                >
                    Create Simulation
                </button>
            </form>
        </div>
    );
};

export default SimulationForm; 