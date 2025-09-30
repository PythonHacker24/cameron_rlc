import React, { useState, useEffect, useRef, useCallback } from 'react';
import { InvertedPendulum } from '@/lib/InvertedPendulum';
import { PIDController } from '@/lib/PIDController';
import PendulumCanvas from '@/components/PendulumCanvas';
import LiveGraph from '@/components/LiveGraph';

interface DataPoint {
  time: number;
  value: number;
}

export default function Home() {
  // Simulation instances (using refs to persist across renders)
  const pendulumRef = useRef<InvertedPendulum | null>(null);
  const controllerRef = useRef<PIDController | null>(null);
  
  // Physical parameters (defined first so they can be used in initial state)
  const [initialAngle, setInitialAngle] = useState(0.1); // Initial angle in radians
  
  // State for visualization
  const [cartPosition, setCartPosition] = useState(0);
  const [pendulumAngle, setPendulumAngle] = useState(initialAngle);
  const [isRunning, setIsRunning] = useState(false);
  const [controllerEnabled, setControllerEnabled] = useState(true);
  
  // PID gains
  const [kp, setKp] = useState(100);
  const [ki, setKi] = useState(1);
  const [kd, setKd] = useState(50);
  
  // Physical parameters (continued)
  const [cartMass, setCartMass] = useState(1.0);
  const [pendulumMass, setPendulumMass] = useState(0.1);
  const [airResistance, setAirResistance] = useState(0.0);
  
  // Display metrics
  const [currentForce, setCurrentForce] = useState(0);
  const [angleDisplay, setAngleDisplay] = useState(initialAngle * (180 / Math.PI));
  
  // Graph data
  const [angleHistory, setAngleHistory] = useState<DataPoint[]>([]);
  const [anglePositionData, setAnglePositionData] = useState<DataPoint[]>([]);
  const startTimeRef = useRef<number>(0);
  
  // Animation frame ref
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // Initialize simulation
  useEffect(() => {
    pendulumRef.current = new InvertedPendulum(1.0, 0.1, 1.0, initialAngle);
    controllerRef.current = new PIDController(kp, ki, kd);
    
    // Update display when initial angle changes
    setPendulumAngle(initialAngle);
    setAngleDisplay(initialAngle * (180 / Math.PI));
  }, [initialAngle]);

  // Update PID gains
  useEffect(() => {
    if (controllerRef.current) {
      controllerRef.current.setGains(kp, ki, kd);
    }
  }, [kp, ki, kd]);

  // Update physical parameters
  useEffect(() => {
    if (pendulumRef.current) {
      pendulumRef.current.setMasses(cartMass, pendulumMass);
      pendulumRef.current.setAirResistance(airResistance);
    }
  }, [cartMass, pendulumMass, airResistance]);

  // Simulation loop
  const simulate = useCallback((currentTime: number) => {
    if (!pendulumRef.current || !controllerRef.current) return;

    const dt = lastTimeRef.current === 0 ? 0.016 : (currentTime - lastTimeRef.current) / 1000;
    lastTimeRef.current = currentTime;

    // Clamp dt to prevent instability
    const clampedDt = Math.min(dt, 0.033); // Max 30fps

    // Get current state
    const state = pendulumRef.current.getState();
    
    // PID control: target angle is 0 (upright)
    let force = 0;
    if (controllerEnabled) {
      force = controllerRef.current.calculate(0, state.pendulumAngle, currentTime / 1000);
    }
    
    // Update physics
    pendulumRef.current.update(force, clampedDt);
    
    // Update display state
    const newState = pendulumRef.current.getState();
    setCartPosition(newState.cartPosition);
    setPendulumAngle(newState.pendulumAngle);
    setCurrentForce(force);
    setAngleDisplay(newState.pendulumAngle * (180 / Math.PI));
    
    // Update graph data
    const elapsedTime = (currentTime - startTimeRef.current) / 1000;
    const angleDegrees = newState.pendulumAngle * (180 / Math.PI);
    
    setAngleHistory(prev => [
      ...prev,
      { time: elapsedTime, value: angleDegrees }
    ]);
    
    setAnglePositionData(prev => [
      ...prev,
      { time: newState.cartPosition, value: angleDegrees }
    ]);

    // Continue animation
    if (isRunning) {
      animationFrameRef.current = requestAnimationFrame(simulate);
    }
  }, [isRunning, controllerEnabled]);

  // Control simulation
  useEffect(() => {
    if (isRunning) {
      lastTimeRef.current = 0;
      startTimeRef.current = performance.now();
      animationFrameRef.current = requestAnimationFrame(simulate);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRunning, simulate]);

  const handleStart = () => {
    setIsRunning(true);
  };

  const handleStop = () => {
    setIsRunning(false);
  };

  const handleReset = () => {
    setIsRunning(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    pendulumRef.current?.reset(initialAngle);
    controllerRef.current?.reset();
    
    const state = pendulumRef.current?.getState();
    if (state) {
      setCartPosition(state.cartPosition);
      setPendulumAngle(state.pendulumAngle);
      setCurrentForce(0);
      setAngleDisplay(state.pendulumAngle * (180 / Math.PI));
    }
    
    // Clear graph data
    setAngleHistory([]);
    setAnglePositionData([]);
    lastTimeRef.current = 0;
    startTimeRef.current = 0;
  };

  const handlePerturbation = () => {
    if (pendulumRef.current) {
      pendulumRef.current.pendulumAngle += 0.3; // Add disturbance
    }
  };

  const toggleController = () => {
    const newState = !controllerEnabled;
    setControllerEnabled(newState);
    
    // Reset controller when enabling to avoid integral windup
    if (newState && controllerRef.current) {
      controllerRef.current.reset();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Inverted Pendulum Simulation
          </h1>
          <p className="text-slate-400 text-lg">
            Real-time physics simulation with PID control
          </p>
        </div>

        {/* Visualization */}
        <div className="mb-8">
          <div className="flex justify-center mb-6">
            <div className="w-full max-w-4xl">
              <PendulumCanvas
                cartPosition={cartPosition}
                pendulumAngle={pendulumAngle}
                scale={60}
              />
            </div>
          </div>
          
          {/* Graphs */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="flex justify-center">
              <div className="w-full">
                <LiveGraph
                  data={angleHistory}
                  title="Angle vs Time"
                  yLabel="Angle (degrees)"
                  xLabel="Time (s)"
                  color="#60a5fa"
                  maxDataPoints={500}
                  yMin={-180}
                  yMax={180}
                />
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className="w-full">
                <LiveGraph
                  data={anglePositionData}
                  title="Angle vs Cart Position"
                  yLabel="Angle (degrees)"
                  xLabel="Position (m)"
                  color="#34d399"
                  maxDataPoints={500}
                  yMin={-180}
                  yMax={180}
                  xMin={-5}
                  xMax={5}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Controls and Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Control Panel */}
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h2 className="text-2xl font-semibold mb-4 text-blue-400">Control Panel</h2>
            
            <div className="space-y-4">
              <div className="flex gap-3">
                <button
                  onClick={handleStart}
                  disabled={isRunning}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
                >
                  Start
                </button>
                <button
                  onClick={handleStop}
                  disabled={!isRunning}
                  className="flex-1 bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-800 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
                >
                  Pause
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
                >
                  Reset
                </button>
              </div>
              
              <button
                onClick={handlePerturbation}
                disabled={!isRunning}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
              >
                Add Disturbance
              </button>
              
              <button
                onClick={toggleController}
                className={`w-full font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 ${
                  controllerEnabled
                    ? 'bg-orange-600 hover:bg-orange-700'
                    : 'bg-teal-600 hover:bg-teal-700'
                }`}
              >
                {controllerEnabled ? 'Disable Controller' : 'Enable Controller'}
              </button>
            </div>
          </div>

          {/* Metrics */}
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h2 className="text-2xl font-semibold mb-4 text-purple-400">System Metrics</h2>
            
            <div className="space-y-3">
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Pendulum Angle</div>
                <div className="text-3xl font-bold text-blue-400">
                  {angleDisplay.toFixed(2)}°
                </div>
              </div>
              
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Cart Position</div>
                <div className="text-3xl font-bold text-green-400">
                  {cartPosition.toFixed(3)} m
                </div>
              </div>
              
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Control Force</div>
                <div className="text-3xl font-bold text-purple-400">
                  {currentForce.toFixed(2)} N
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PID Tuning */}
        <div className={`bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 transition-opacity duration-300 ${
          controllerEnabled ? 'opacity-100' : 'opacity-50'
        }`}>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-blue-400">PID Controller Tuning</h2>
            {!controllerEnabled && (
              <span className="text-sm bg-orange-600/20 text-orange-400 px-3 py-1 rounded-full border border-orange-600/30">
                Controller Disabled
              </span>
            )}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Proportional (Kp): <span className="text-blue-400 font-bold">{kp}</span>
              </label>
              <input
                type="range"
                min="0"
                max="200"
                step="1"
                value={kp}
                onChange={(e) => setKp(Number(e.target.value))}
                disabled={!controllerEnabled}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Integral (Ki): <span className="text-green-400 font-bold">{ki}</span>
              </label>
              <input
                type="range"
                min="0"
                max="20"
                step="0.1"
                value={ki}
                onChange={(e) => setKi(Number(e.target.value))}
                disabled={!controllerEnabled}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Derivative (Kd): <span className="text-purple-400 font-bold">{kd}</span>
              </label>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={kd}
                onChange={(e) => setKd(Number(e.target.value))}
                disabled={!controllerEnabled}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
          </div>
        </div>

        {/* Physical Parameters */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 mt-6">
          <h2 className="text-2xl font-semibold mb-6 text-green-400">Physical Parameters</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Cart Mass (kg): <span className="text-green-400 font-bold">{cartMass.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="5"
                step="0.1"
                value={cartMass}
                onChange={(e) => setCartMass(Number(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Pendulum Mass (kg): <span className="text-blue-400 font-bold">{pendulumMass.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.01"
                max="1"
                step="0.01"
                value={pendulumMass}
                onChange={(e) => setPendulumMass(Number(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Air Resistance: <span className="text-purple-400 font-bold">{airResistance.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={airResistance}
                onChange={(e) => setAirResistance(Number(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Initial Angle (°): <span className="text-orange-400 font-bold">{(initialAngle * 180 / Math.PI).toFixed(1)}</span>
              </label>
              <input
                type="range"
                min="-180"
                max="180"
                step="1"
                value={initialAngle * 180 / Math.PI}
                onChange={(e) => setInitialAngle(Number(e.target.value) * Math.PI / 180)}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
          </div>
          
          <div className="mt-4 text-sm text-slate-400">
            <p>
              💡 <strong>Tip:</strong> Higher cart mass makes the system more stable but harder to control. 
              Higher pendulum mass increases instability. Air resistance dampens motion. Initial angle determines starting position (0° = upright).
            </p>
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-8 text-center text-slate-400 text-sm">
          <p>
            {controllerEnabled
              ? 'Use the sliders to tune the PID controller and observe how it affects the pendulum balance. The controller outputs force to keep the pendulum upright at 0°.'
              : 'Controller is disabled. The pendulum will fall freely under gravity with no control force applied.'}
          </p>
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          background: #3b82f6;
          cursor: pointer;
          border-radius: 50%;
          transition: all 0.2s;
        }
        
        .slider::-webkit-slider-thumb:hover {
          background: #2563eb;
          transform: scale(1.2);
        }
        
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          background: #3b82f6;
          cursor: pointer;
          border-radius: 50%;
          border: none;
          transition: all 0.2s;
        }
        
        .slider::-moz-range-thumb:hover {
          background: #2563eb;
          transform: scale(1.2);
        }
      `}</style>
    </div>
  );
}
