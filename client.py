"""
client.py
----------
Main entry point for the Racing Simulator.

Two operating modes:
  --mode manual  : Human drives with arrow keys; observations + actions are
                   recorded to data/ for later training.
  --mode ai      : Trained model drives autonomously; loads driving_model.pt
                   and norm_stats.npz from models/.

Usage:
  # Collect driving data (arrow keys control the car)
  python client.py --mode manual

  # Let the trained AI drive
  python client.py --mode ai

  # Full options
  python client.py --mode manual --config config.json --port 5004
  python client.py --mode ai --model models/driving_model.pt --norm-stats models/norm_stats.npz
"""

import argparse
import json

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.string_log_channel import StringLogChannel

from ai_model import DrivingModel, load_normalization_stats, normalize
from data_collector import DataCollector
from input_manager import InputManager


# ===========================================================================
# Unity connection
# ===========================================================================


def load_config(config_path: str = "config.json") -> dict:
    """Load and return the agent configuration dict."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def connect_to_unity(config: dict, port: int = 5004) -> tuple:
    """
    Open a connection to the running Unity simulation.

    Returns:
        env           : UnityEnvironment
        behavior_name : str  (name of the first registered behavior)
    """
    print(f"[Client] Connecting to Unity on port {port}...")

    # StringLogChannel lets us send the agent config JSON to Unity
    string_channel = StringLogChannel()

    env = UnityEnvironment(
        file_name=None,  # simulator must already be running
        base_port=port,
        side_channels=[string_channel],
        additional_args=["--config-path", "config.json"],
    )
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        raise RuntimeError("No behavior found in the Unity environment. Is the simulator running?")

    behavior_name = behavior_names[0]
    print(f"[Client] Connected. Behavior: '{behavior_name}'")
    return env, behavior_name


# ===========================================================================
# Observation parsing
# ===========================================================================


def parse_observation(decision_steps, agent_index: int = 0) -> np.ndarray:
    """
    Extract the flat observation array for a single agent.

    The ray cast sensor returns observations as a list of arrays.
    We take the first observation array for the requested agent.

    Args:
        decision_steps: DecisionSteps object from env.get_steps()
        agent_index:    index into the decision_steps agents list

    Returns:
        1-D float32 numpy array of shape (nbRay + 1,)
    """
    # decision_steps.obs is a list of arrays, one per sensor
    # First sensor: ray distances + speed, already concatenated by Unity
    obs = decision_steps.obs[0][agent_index]
    return obs.astype(np.float32).flatten()


# ===========================================================================
# Manual driving mode
# ===========================================================================


def run_manual_mode(
    env: UnityEnvironment,
    behavior_name: str,
    input_manager: InputManager,
    data_collector: DataCollector,
) -> None:
    """
    Main loop for manual driving with data collection.

    Controls (arrow keys):
      LEFT / RIGHT  — steer
      UP            — accelerate
      DOWN          — brake

    Press Ctrl+C to stop and save data.
    """
    print("\n[Manual] Driving mode started.")
    print("[Manual] Use arrow keys to control the car. Press Ctrl+C to stop.\n")

    step = 0
    try:
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) == 0:
                env.step()
                continue

            obs = parse_observation(decision_steps, agent_index=0)
            steering, acceleration = input_manager.get_action()

            data_collector.record(obs, steering, acceleration)

            action = ActionTuple(continuous=np.array([[steering, acceleration]], dtype=np.float32))
            env.set_actions(behavior_name, action)
            env.step()
            step += 1

            if len(terminal_steps) > 0:
                print(f"[Manual] Episode ended at step {step}. Resetting...")
                env.reset()

    except KeyboardInterrupt:
        print(f"\n[Manual] Stopped after {step} steps.")
    finally:
        data_collector.close()


# ===========================================================================
# AI driving mode
# ===========================================================================


def run_ai_mode(
    env: UnityEnvironment,
    behavior_name: str,
    model: DrivingModel,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> None:
    """
    Main loop for autonomous AI driving.

    Observations are normalised before being passed to the model.
    Press Ctrl+C to stop.
    """
    print("\n[AI] Autonomous driving mode started.")
    print("[AI] Press Ctrl+C to stop.\n")

    step = 0
    try:
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) == 0:
                env.step()
                continue

            obs = parse_observation(decision_steps, agent_index=0)
            obs_norm = normalize(obs, norm_mean, norm_std)
            steering, acceleration = model.predict(obs_norm)

            action = ActionTuple(continuous=np.array([[steering, acceleration]], dtype=np.float32))
            env.set_actions(behavior_name, action)
            env.step()
            step += 1

            if len(terminal_steps) > 0:
                print(f"[AI] Episode ended at step {step}. Resetting...")
                env.reset()

    except KeyboardInterrupt:
        print(f"\n[AI] Stopped after {step} steps.")


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Racing Simulator Client — manual driving or AI driving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --mode manual
  python client.py --mode ai
  python client.py --mode ai --model models/driving_model.pt --norm-stats models/norm_stats.npz
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "ai"],
        required=True,
        help="'manual' to drive and collect data, 'ai' to run the trained model",
    )
    parser.add_argument("--config", default="config.json", help="Path to agent config JSON")
    parser.add_argument("--model", default="models/driving_model.pt", help="Path to trained model (.pt)")
    parser.add_argument("--norm-stats", default="models/norm_stats.npz", help="Path to normalization stats (.npz)")
    parser.add_argument("--port", type=int, default=5004, help="Unity environment port")
    parser.add_argument("--data-dir", default="data", help="Directory for collected CSV data (manual mode)")
    args = parser.parse_args()

    config = load_config(args.config)
    env = None
    input_manager = None
    data_collector = None

    try:
        env, behavior_name = connect_to_unity(config, port=args.port)

        if args.mode == "manual":
            input_manager = InputManager()
            input_manager.start()
            data_collector = DataCollector(data_dir=args.data_dir)
            run_manual_mode(env, behavior_name, input_manager, data_collector)

        else:  # ai
            model = DrivingModel.load(args.model)
            norm_mean, norm_std = load_normalization_stats(args.norm_stats)
            run_ai_mode(env, behavior_name, model, norm_mean, norm_std)

    finally:
        if input_manager is not None:
            input_manager.stop()
        if env is not None:
            env.close()
            print("[Client] Environment closed.")


if __name__ == "__main__":
    main()
