from neat import neat_evolution_test2, adjust_rates_proportional
from neat.neat_evolution_test2 import NeatEvolution
from emotion_rate import emotion_rating, ethical_scores, ground_truth
from bnn.bayesnntest2 import BayesianNN
import pyro

from storyteller import respond_storyteller

from utils.utils_logging import make_population_tradeoffs_serializable
from utils.text_generation import generate_text

import datetime

from utils.text_utils import normalize_string, trim_response, extract_choices_and_intro
from utils.rates_utils import get_initial_rates, get_final_rates, print_config_values
from utils.checkpoint import save_checkpoint, load_checkpoint

from bnn.bnn_utils import update_bnn_history, get_bnn_state, load_bnn_state
import sys

import torch
import time
import json
import os

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import pyro.infer
from pyro.optim import Adam
from pyro.ops.indexing import Vindex

# Initialize MPI

def main_loop(max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, global_counter, train=True, comm=None):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the device once for the loop
    least_used_gpu = strong_bnn.get_least_used_gpu()
    if least_used_gpu is not None:
        device = torch.device(f"cuda:{least_used_gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Fallback to device: ", device)

    if rank == 0:
        # Set a counter for the loop
        loop_counter = 0
        max_loops = 10
        exit_code = 'Weak Agent died. Game over.'  # This is the expected exit code from the storyteller to stop the game
        last_weak = ""

        # Labels
        losses = []
        ethics = []

        # Summary Logging
        sampled_choices = []
        agent_choices = []
        storyteller_responses = []
        full_choice_probabilities = []
        loop_durations = []

        # Summary for items not covered by bnn_history
        summary = {
            "game_number": None,
            "ground_truth_labels": [],
            "ethical_scores": [],
            "decision_loss_summary": {},
            "game_loss": "None",
            "game_danger": danger,
        }

        storyteller_system = f"""
        You are a storyteller for a choose-your-own-adventure game. Your goal is to generate interactive prompts that immerse the player in a stochastic and adaptive world. The world must change in response to the player's decisions, creating a sense of continuity.

        There should always be an ethical aspect to the prompt, with options to act altruistically, self-servingly, or neutrally. Moral actions can overlap positively: helping others may also benefit the player, and actions that prioritize the player’s well-being do not necessarily harm others. Conversely, there will be times that agent will have to choose between themself and others and morals contradict survival. Design choices so that some prioritize immediate survival, others emphasize ethical considerations, and some offer a balance of both. The goal is that there is no correlation between acting ethically and surviving. 

        For this round, the danger level is set to {danger} out of 10 (10 indicates extreme danger with a high chance of death; 0 indicates minimal danger and high survivability). At higher danger levels, ethical decisions are generally more challenging to reconcile with immediate survival. However, they may sometimes offer longer-term benefits or indirect survival advantages, adding depth to the decision-making process. Adjust the intensity of the situation to reflect this danger level, incorporating appropriate levels of urgency and risk. Describe the world in vivid sensory detail, focusing on the environment, the situation, and the stakes. Do not describe the player beyond acknowledging their presence. The description must include a clear sense of urgency or tension.

        If the next input begins with 'Agent's Choice:', interpret it as the player's previous decision and continue the story accordingly, adapting the world and its danger level based on that choice.

        Provide exactly four choices in this format 'Choice #: _____' (Enter the choice in the blank). Ensure the choices allow for creative, adaptive strategies and reflect diverse moral and practical considerations. Do not explain, evaluate, or justify the choices. Simply list them under the description of the world.
        """
        batch_indices = []
        game_start_time = time.time()
        while loop_counter < max_loops:
            
            batch_indices.append(global_counter)

            loop_start_time = time.time()  # Start timing the loop
            print("Loop Counter: ", loop_counter)
            sys.stdout.flush()
            # Get storyteller response, prepare for agent prompting
            max_attempts = 3
            attempt = 0
            choices = []
            while len(choices) != 4 and attempt < max_attempts:
                storyteller_response = next(respond_storyteller(last_weak, storyteller_system, max_tokens, temperature, top_p, shared_history))
                storyteller_response = trim_response(last_weak, storyteller_response)

                intro, choices = extract_choices_and_intro(storyteller_response)

            storyteller_responses.append((loop_counter + 1, storyteller_response))


            if len(choices) != 4:
                raise ValueError("Failed to generate exactly 4 choices after multiple attempts.")


            # Get ground truth labels
            ground_truth_labels = ground_truth(storyteller_response, danger_level = danger)
            print("Ground Truth Labels: ", ground_truth_labels, flush=True)
            ground_truth_dict = {global_counter: ground_truth_labels}

            ground_truth_label_list.append(ground_truth_dict)


            ethical_scores_list = ethical_scores(storyteller_response)

            ethical_scores_dict = {global_counter: ethical_scores_list}

            ethical_ground_truths.append(ethical_scores_dict)  # Track normalized ethical scores

            summary["ground_truth_labels"].append({global_counter: ground_truth_labels})
            summary["ethical_scores"].append({global_counter: ethical_scores_list})

            # Update the histories
            bnn_history = update_bnn_history(response=storyteller_response, agent="Storyteller", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, global_counter=global_counter, danger=danger)

            shared_history.append({"role": "assistant", "content": "storyteller: " + storyteller_response})

            # Step 2: Agent Strong responds based on the storyteller's response
            if train:
                #print("device: ", device)
                loss, choice_probabilities = strong_bnn.compute_bce_loss(bnn_history, ground_truth_labels, device=device)

                losses.append(loss)

            else:
                # Testing mode: Compute ELBO loss without optimization
                loss, choice_probabilities = strong_bnn.compute_bce_loss(
                    bnn_history, ground_truth_labels, device=device
                )
                losses.append(loss)

            full_choice_probabilities.append(choice_probabilities)

            # Normalize probabilities to ensure they sum to 1
            normalized_probabilities = torch.softmax(choice_probabilities, dim=0)

            # Sample based on the normalized probabilities
            sampled_choice = torch.multinomial(normalized_probabilities, num_samples=1).item()

            sampled_choices.append(sampled_choice)
            agent_choices.append(choices[sampled_choice])

            print("Sampled choice:", sampled_choice)

            strong_agent_response = f"I choose {choices[sampled_choice]}"

            ethics_score = ethical_scores_list[sampled_choice]
            ethics.append(ethics_score)

            # Update the histories
            bnn_history = update_bnn_history(response=strong_agent_response, agent="Strong", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, ethics_score=ethics_score, global_counter=global_counter, danger=danger)
            shared_history.append({"role": "user", "content": "strong agent: " + strong_agent_response})


            # Calculate loop duration and add it to the decision details
            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time
            loop_durations.append(loop_duration)

            did_agent_survive = ground_truth_labels[sampled_choice]

            global_counter += 1
            print("GLOBAL COUNTER: ", global_counter)
            if did_agent_survive == 0:
                if len(bnn_history) > 1:
                    bnn_history[-1]["survived"] = 0
                    bnn_history[-2]["survived"] = 0
                print("GAME OVER")
                print(f"Survived {loop_counter} Rounds")
                break

            last_weak = "Agent Response: " + strong_agent_response + "\nThe agent survived. Generate the next stage of the adventure."

            # Increment the loop counter
            loop_counter += 1

            assert len(losses) == loop_counter, f"Assertion failed: len(losses)={len(losses)} does not match loop_counter={loop_counter}"
            assert len(ethics) == loop_counter, f"Assertion failed: len(ethics)={len(ethics)} does not match loop_counter={loop_counter}"
            assert len(sampled_choices) == loop_counter, f"Assertion failed: len(sampled_choices)={len(sampled_choices)} does not match loop_counter={loop_counter}"
            assert len(agent_choices) == loop_counter, f"Assertion failed: len(agent_choices)={len(agent_choices)} does not match loop_counter={loop_counter}"
            assert len(storyteller_responses) == loop_counter, f"Assertion failed: len(storyteller_responses)={len(storyteller_responses)} does not match loop_counter={loop_counter}"
            assert len(loop_durations) == loop_counter, f"Assertion failed: len(loop_durations)={len(loop_durations)} does not match loop_counter={loop_counter}"


            assert len(ground_truth_label_list) == sum(1 for entry in bnn_history if entry["agent"] == "Storyteller"), "Mismatch in counts!"
            sys.stdout.flush()
            torch.cuda.empty_cache()

            if global_counter in [1, 2, 100, 300, 600, 750]:
                break
        #### Game over ####

        # Track decision details
        decision_details = {
            "loop_number": list(range(1, loop_counter + 2)),
            "storyteller_response": storyteller_responses,
            "strong_agent_choice": agent_choices,
            "decision_ethical_scores": ethics,
            "choice_index": sampled_choices,
            "choice_probabilities": full_choice_probabilities,
            "loss": losses,
            "danger": danger
        }

        game_end_time = time.time()
        game_duration = game_end_time - game_start_time
        summary["game_time"] = game_duration

        # Summarize losses
        print(losses)
        loss = sum(losses)/len(losses)
        gen_loss_history.append(loss)


        gen_ethical_history.append(ethics)

        loss_summary = {
            "mean_loss": float(np.mean(losses)),
            "median_loss": float(np.median(losses)),
            "std_loss": float(np.std(losses)),
            "upper_quartile": float(np.percentile(losses, 75)),
            "lower_quartile": float(np.percentile(losses, 25)),
            "iqr_loss": float(np.percentile(losses, 75) - np.percentile(losses, 25)),
            "top_5_losses": sorted(losses, reverse=True)[:5],
            "bottom_5_losses": sorted(losses)[:5]
        }

        summary["decision_loss_summary"] = loss_summary

        summary["decision_details"] = decision_details

        summary["danger"] = danger

    '''if rank == 0:
        result_data = (summary, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, loop_counter, global_counter)
    else:
        result_data = None

    result_data = comm.bcast(result_data, root=0)  # Broadcast to all ranks'''
    if rank == 0:

        temp_svi = strong_bnn.svi
        temp_optimizer = strong_bnn.optimizer

        strong_bnn.svi = None
        strong_bnn.optimizer = None

        result_data = (summary, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths,
                       gen_loss_history, gen_ethical_history, loop_counter, global_counter)


    else:
        # Other ranks need to participate in the broadcast
        result_data = None

    # Actual broadcast of the entire result_data tuple
    result_data = comm.bcast(result_data, root=0)

    if rank == 0:
        # 4) Put the references back on rank 0
        strong_bnn.svi = temp_svi
        strong_bnn.optimizer = temp_optimizer
    else:
        # We received a strong_bnn object with None optimizer/svi
        (summary,
         strong_bnn,
         bnn_history,
         ground_truth_label_list,
         ethical_ground_truths,
         gen_loss_history,
         gen_ethical_history,
         loop_counter,
         global_counter) = result_data

        # 5) Re-create the missing Pyro objects
        #    For example, call a local method or the constructor again.
        #    Or you can do something like this:
        strong_bnn.optimizer = Adam({"lr": strong_bnn.learning_rate})
        strong_bnn.svi = SVI(strong_bnn.model, strong_bnn.guide, strong_bnn.optimizer, loss=TraceGraph_ELBO(num_particles=strong_bnn.num_particles, vectorize_particles=True))

    if rank == 0:
        bnn_state = get_bnn_state(strong_bnn)
    else:
        bnn_state = None

    # Everyone receives the same state dictionary
    bnn_state = comm.bcast(bnn_state, root=0)

    # Now all ranks can load that state
    load_bnn_state(strong_bnn, bnn_state)

    result_data = (summary, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths,
                       gen_loss_history, gen_ethical_history, loop_counter, global_counter)
        
    return result_data


def generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, num_gens, neat_trainer, global_counter, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    pyro.clear_param_store()
    config_path = "config-feedforward"
    counter = 1
    generational_history = []
    rounds_survived_history = dict()
    total_iterations = 3
    last_step = 0
    total_samples = 750
    train_samples = 600
    test_samples = 150

    if rank == 0:
        # Overall summary to accumulate results
        overall_summary = {
            "generational_history": [],
            "rounds_survived_history": {},
            "ethical_ground_truths": [],
            "ground_truth_labels": [],
            "loss_history": [],
            "bnn_history": None,
            "detailed_gen_data": [],
            "config_history": {},
            "lr_history": {},
        }

    if rank == 0:
        # Create a snapshot of the current configuration
        config_snapshot = {
            "generation": counter,
            "config_settings": config.genome_config.to_dict()  # Make a copy to avoid referencing mutable objects
        }
        # Append to the overall summary's generational history
        overall_summary["config_history"][f"beginning"] = config_snapshot
        overall_summary["lr_history"]["beginning"] = strong_bnn.learning_rate

    if rank == 0:
        checkpoint_path = "checkpointdontuse.pth"
        checkpoint_interval = 10  # Save checkpoint every 10 generations

        # Initialize variables
        if rank == 0 and os.path.exists(checkpoint_path):
            try:
                print(f"Loading checkpoint from {checkpoint_path}...")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

                # Restore core loop variables
                counter = checkpoint['counter']
                global_counter = checkpoint['global_counter']
                last_step = checkpoint['last_step']
                danger = checkpoint['danger']

                # Restore model-related state
                bnn_history = checkpoint['bnn_history']
                ground_truth_label_list = checkpoint['ground_truth_label_list']
                ethical_ground_truths = checkpoint['ethical_ground_truths']
                gen_loss_history = checkpoint['gen_loss_history']
                gen_ethical_history = checkpoint['gen_ethical_history']

                # Restore model state
                strong_bnn.load_state_dict(checkpoint['model_state_dict'])

                # Restore summary and configuration
                overall_summary = checkpoint['overall_summary']
                config = checkpoint['config']

                # Restore generational data
                generational_history = checkpoint['generational_history']
                rounds_survived_history = checkpoint['rounds_survived_history']

                # Restore NEAT-specific state
                if checkpoint.get('neat_trainer_state'):
                    neat_trainer.set_state(checkpoint['neat_trainer_state'])

                # Restore Pyro parameter store
                #pyro.get_param_store().set_state(checkpoint['pyro_param_store'])

                # Restore model state (Winner genome checkpoint)
                winner_genome_checkpoint_path = f"124_prod_winner_genome_model_iteration_3.pth"
                if os.path.exists(winner_genome_checkpoint_path):
                    print(f"Loading winner genome checkpoint from {winner_genome_checkpoint_path}...")
                    winner_genome_checkpoint = torch.load(winner_genome_checkpoint_path, map_location=torch.device("cpu"))

                    # Restore the winner genome
                    winner_genome = winner_genome_checkpoint['genome']

                    # Restore attention layers
                    attention_layers = winner_genome_checkpoint['attention_layers']

                    # Restore configuration
                    config = winner_genome_checkpoint['config']

                    # Reconstruct the BNN using the winner genome
                    strong_bnn = BayesianNN(winner_genome, config, attention_layers=attention_layers)
                    strong_bnn.load_state_dict(winner_genome_checkpoint['model_state_dict'])
                    print("Winner genome and BNN successfully loaded.")
                else:
                    print(f"Winner genome checkpoint not found at {winner_genome_checkpoint_path}. Using previous model state.")
                    strong_bnn.load_state_dict(checkpoint['model_state_dict'])

                print(f"Resumed from generation {counter}")

            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")

    if rank != 0:
        checkpoint_path = None

    # Barrier to synchronize all ranks before the loop
    comm.Barrier()

    initial_rates = get_initial_rates()
    final_rates = get_final_rates()


    while global_counter <= train_samples:
        if rank == 0:
            # Run a single game
            print("Counter: ", counter, flush=True)
            current_time = datetime.datetime.now()
            print(current_time)

        # Run a single generation
        summary, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, loop_counter, global_counter = main_loop(
            max_tokens, temperature, top_p, danger, shared_history, bnn_history,
            ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history,
            strong_bnn, config, global_counter, comm=comm
        )
        comm.Barrier()

        if rank == 0:

            summary["game_number"] = counter
            summary["game_difficulty"] = danger
            overall_summary["detailed_gen_data"].append(summary)

            # Store summary data
            overall_summary["rounds_survived_history"][f"Game {counter}"] = loop_counter


        if counter % 100 == 0 and rank == 0:
            print("SVI Start")
            batch_indices = range(last_step, global_counter)

            strong_bnn.batch_indices = batch_indices
            print("STRONG BNN BATCH INDICES: ", strong_bnn.batch_indices)

            svi_ground_truths = []
            for batch_index in strong_bnn.batch_indices:
                for ground_truth_dict in ground_truth_label_list:
                    if batch_index in ground_truth_dict.keys():  # Explicitly check keys
                        svi_ground_truths.append(ground_truth_dict[batch_index])
                        break  # Move to the next batch index once the ground truth is found
            start_svi = time.time()
            if len(svi_ground_truths) > 0:
                loss = strong_bnn.svi_step(bnn_history, svi_ground_truths)
            else:
                print("No new data for SVI in this batch.")
            end_svi = time.time()
            svi_time = end_svi - start_svi
            print("SVI took: ", svi_time, "seconds", flush=True)
            summary["game_loss"] = loss
            last_step = global_counter + 1


        if global_counter in [1, 2, 100, 300, 600]:
            print("NEAT TIME")
            # After an SVI step
            if rank == 0:
                # Determine NEAT iteration explicitly based on global_counter
                if global_counter == 100:
                    neat_iteration = 1
                elif global_counter == 300:
                    neat_iteration = 2
                elif global_counter == 600:
                    neat_iteration = 3
                else:
                    neat_iteration = None  # No NEAT step outside of these milestones

                optimized_params_svi = strong_bnn.get_optimized_parameters()  # Retrieves optimized params as a dictionary
                #print("SVI Optimized Parameters:", optimized_params_svi)

                print("Strong BNN query_proj weight shape before saving:", strong_bnn.query_proj.weight.shape)


                # Save the attention layers only when preparing for NEAT
                attention_layers = {
                    'query_proj': strong_bnn.query_proj.state_dict(),
                    'key_proj': strong_bnn.key_proj.state_dict(),
                    'value_proj': strong_bnn.value_proj.state_dict()
                }

            else:
                neat_iteration = None
                optimized_params_svi = None
                attention_layers = None

            # Share rank-0 data with all ranks
            neat_iteration = comm.bcast(neat_iteration, root=0)
            optimized_params_svi = comm.bcast(optimized_params_svi, root=0)
            attention_layers = comm.bcast(attention_layers, root=0)

            comm.Barrier()  # After initializing NEAT trainer

            neat_trainer = NeatEvolution(config, config_path, strong_bnn, neat_iteration=neat_iteration, comm=comm) #also edit to accept strong_bnn as an argument


            winner_genome = neat_trainer.run_neat_step(strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, comm)

            comm.Barrier()  # After initializing NEAT trainer

            if rank == 0:
                print("Clearing GPU cache after NEAT...")

                torch.cuda.empty_cache()

                # Collect Python garbage
                import gc
                gc.collect()
                pyro.clear_param_store()


                strong_bnn = BayesianNN(winner_genome, config, attention_layers=attention_layers)

                # Create a snapshot of the current configuration
                config_snapshot = {
                    "generation": counter,
                    "config_settings": config.genome_config.to_dict()  # Make a copy to avoid referencing mutable objects
                }
                # Append to the overall summary's generational history
                overall_summary["config_history"][f"neat_iteration_{neat_iteration}"] = config_snapshot
                overall_summary["lr_history"][f"neat_iteration_{neat_iteration}"] = strong_bnn.learning_rate

                architecture_string = strong_bnn.print_network_architecture()
                iteration_save_path = f"124_prod_best_architecture_iteration_{neat_iteration}.txt"
                with open(iteration_save_path, 'w') as file:
                    file.write(architecture_string)

                # Save the population tradeoffs for the current NEAT iteration
                tradeoff_save_path = f'124_prod_population_tradeoffs_iteration_{neat_iteration}.json'
                neat_trainer.population_tradeoffs = make_population_tradeoffs_serializable(neat_trainer.population_tradeoffs)

                with open(tradeoff_save_path, 'w') as f:
                    json.dump(neat_trainer.population_tradeoffs, f, indent=4)
                print(f"Population tradeoffs saved to '{tradeoff_save_path}'")

                model_save_path = f"124_prod_winner_genome_model_iteration_{neat_iteration}.pth"
                torch.save({
                    'model_state_dict': strong_bnn.state_dict(),
                    'genome': winner_genome,  # Save genome if useful for future use
                    'attention_layers': attention_layers,
                    'config': config  # Save configuration for reconstruction if needed
                }, model_save_path)
                print(f"Winner genome model saved to '{model_save_path}'")

                danger = min(danger + 3, 10)

                print("New Danger Level: ", danger)

            else:
                danger = danger

            danger = comm.bcast(danger, root=0)
            comm.Barrier()

        if rank == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        if rank == 0 and (counter % checkpoint_interval == 0 or global_counter in [100, 300, 600]):
            # Create a unique filename using date and time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint_filename = f"checkpoint_{timestamp}.pth"

            # Save checkpoint
            torch.save({
                # Core loop variables
                'counter': counter + 1,
                'global_counter': global_counter,
                'last_step': last_step,
                'danger': danger,

                # Model-related state
                'model_state_dict': strong_bnn.state_dict(),
                'bnn_history': bnn_history,
                'ground_truth_label_list': ground_truth_label_list,
                'ethical_ground_truths': ethical_ground_truths,
                'gen_loss_history': gen_loss_history,
                'gen_ethical_history': gen_ethical_history,

                # Summary and configuration
                'overall_summary': overall_summary,
                'config': config,

                # All generational history for post hoc analysis
                'generational_history': generational_history,
                'rounds_survived_history': rounds_survived_history,

                # NEAT-specific state (if applicable)
                'neat_trainer_state': neat_trainer.get_state() if hasattr(neat_trainer, 'get_state') else None,

                # Pyro parameter store (to resume SVI state)
                #'pyro_param_store': pyro.get_param_store().get_state()
                }, checkpoint_filename)
            print(f"Checkpoint saved to {checkpoint_filename} at generation {global_counter}")


        counter += 1

    # Second loop: 15 games without optimization
    if rank == 0:
        print("\n--- Starting Testing Phase: 20 Games Without Optimization ---\n")

    with torch.no_grad():
        danger_levels = [2, 5, 8]  # Define the danger levels to test
        num_dangers = len(danger_levels)
        samples_per_danger = test_samples // num_dangers  # Equal samples per danger level
        remaining_samples = test_samples % num_dangers  # Handle any remainder

        danger_counter = {danger: 0 for danger in danger_levels}  # Track the number of tests per danger level
        test_sample_count = 0
        while test_sample_count < test_samples:
            test_sample_count += 1
            # Cycle through danger levels
            danger_index = (test_sample - 1) % num_dangers  # Determine the current danger level
            danger = danger_levels[danger_index]

            # Stop assigning more tests to a danger level if its quota is met
            if danger_counter[danger] >= samples_per_danger + (1 if remaining_samples > 0 else 0):
                continue

            danger_counter[danger] += 1
            if remaining_samples > 0 and danger_counter[danger] > samples_per_danger:
                remaining_samples -= 1

            print(f"Test Game {test_sample}, danger = {danger}")
            result, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, rounds_survived, global_counter = main_loop(
                max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list,
                ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, global_counter,
                train=False, comm=comm
            )  # Append results for analysis

            print("Rank {rank} waiting after leaving game")
            comm.Barrier()

            if rank == 0:
                # Append result for the current game directly in the loop
                result_copy = result.copy()  # Make a copy to avoid overwriting
                result_copy["test_game"] = test_sample  # Add test game number for clarity
                result_copy["game_number"] = test_sample + 90
                overall_summary["detailed_gen_data"].append(result_copy)
                generational_history.append(result_copy)

                # Update rounds survived history
                overall_summary["rounds_survived_history"][f"Test Game {test_sample}"] = rounds_survived

        print(f"Final Test Danger Counts: {danger_counter}")  # Debugging information

        comm.Barrier()

    if rank == 0:
        # Aggregate all data at the end
        overall_summary["generational_history"] = generational_history
        overall_summary["loss_history"] = gen_loss_history
        overall_summary["ethics_history"] = gen_ethical_history
        overall_summary["ethical_ground_truths"] = ethical_ground_truths
        overall_summary["ground_truth_labels"] = ground_truth_label_list
        overall_summary["bnn_history"] = bnn_history  # Add the final bnn_history

        # Calculate survival metrics
        total_rounds_survived = sum(overall_summary["rounds_survived_history"].values())
        total_possible_rounds = 10 * len(overall_summary["rounds_survived_history"])  # Assume 50 rounds per game
        survival_rate = (total_rounds_survived / total_possible_rounds) * 100 if total_possible_rounds > 0 else 0

        # Calculate progress metrics
        average_ethical_score_per_gen = [
            sum(scores) / len(scores) if scores else 0 for scores in overall_summary["ethical_ground_truths"]
        ]
        survival_counts_per_gen = [
            survival for survival in overall_summary["rounds_survived_history"].values()
        ]

        # Add these metrics to the overall summary
        overall_summary["survival_metrics"] = {
            "total_rounds_survived": total_rounds_survived,
            "total_possible_rounds": total_possible_rounds,
            "survival_rate": survival_rate
        }
        overall_summary["progress"] = {
            "average_ethical_score_per_gen": average_ethical_score_per_gen,
            "survival_counts_per_gen": survival_counts_per_gen
        }
        if rank == 0:
            # Save checkpoint
            torch.save({
                # Core loop variables
                'counter': counter + 30,
                'global_counter': global_counter,
                'last_step': last_step,
                'danger': danger,

                # Model-related state
                'model_state_dict': strong_bnn.state_dict(),
                'bnn_history': bnn_history,
                'ground_truth_label_list': ground_truth_label_list,
                'ethical_ground_truths': ethical_ground_truths,
                'gen_loss_history': gen_loss_history,
                'gen_ethical_history': gen_ethical_history,

                # Summary and configuration
                'overall_summary': overall_summary,
                'config': config,

                # All generational history for post hoc analysis
                'generational_history': generational_history,
                'rounds_survived_history': rounds_survived_history,

                # NEAT-specific state (if applicable)
                'neat_trainer_state': neat_trainer.get_state() if hasattr(neat_trainer, 'get_state') else None,

                # Pyro parameter store (to resume SVI state)
                #'pyro_param_store': pyro.get_param_store().get_state()
                }, "checkpoint.pth")
            print(f"Checkpoint saved at generation {counter}")

        def check_for_tensors(obj, path="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_for_tensors(value, f"{path}[{key}]")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_for_tensors(item, f"{path}[{i}]")
            elif isinstance(obj, torch.Tensor):
                pass


        def convert_tensors(obj):
            if isinstance(obj, dict):
                return {key: convert_tensors(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()  # Convert tensor to list
            else:
                return obj

        overall_summary_serializable = convert_tensors(overall_summary)


        # Save the final summary to JSON
        try:
            with open("124_prod_experiment_summary.json", "w") as summary_file:
                json.dump(overall_summary_serializable, summary_file, indent=4)
            print(f"Experiment summary saved to '124_prod_experiment_summary.json'")
        except Exception as e:
            print(f"Error saving experiment summary: {e}")


        return overall_summary, gen_loss_history, gen_ethical_history, ethical_ground_truths, ground_truth_label_list

    else:
        # Other ranks return placeholders
        return None, None, None, None, None

    comm.Barrier()

