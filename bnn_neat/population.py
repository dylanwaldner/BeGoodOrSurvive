"""Implements the core evolution algorithm."""
from utils.rates_utils import get_initial_rates, get_final_rates, print_config_values
from bnn_neat.math_util import mean
from bnn_neat.reporting import ReporterSet
from bnn_neat.genome import DefaultGenome
import json
import datetime
from mpi4py import MPI
import pickle
import json

class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        print_config_values(self.config)
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def adjust_compatibility_threshold(self, config, species_count, desired_min_species, desired_max_species, adjustment_step=0.5):
        """
        Adjusts the compatibility threshold to maintain the desired number of species.
        """
        if species_count < desired_min_species:
            # Too few species; decrease the threshold to encourage more speciation
            self.config.species_set_config.compatibility_threshold -= adjustment_step
            print(f"Decreasing compatibility threshold to {config.species_set_config.compatibility_threshold}")
        elif species_count > desired_max_species:
            # Too many species; increase the threshold to encourage fewer speciation
            self.config.species_set_config.compatibility_threshold += adjustment_step
            print(f"Increasing compatibility threshold to {config.species_set_config.compatibility_threshold}")


    def set_stagnation_limit(self, new_limit):
        """
        Sets a new stagnation limit in the configuration.
        """
        self.config.stagnation_config.species_fitness_func_args['max_stagnation'] = new_limit
        print(f"Set stagnation limit to {new_limit}")


    def run(self, fitness_function, n=None, neat_iteration="NoneSet", comm=None):
        rank = comm.Get_rank()
        size = comm.Get_size()

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        evolution_data = {"generations": []}
        k = 0

        while n is None or k < n:
            current_time = datetime.datetime.now()
            print(current_time)
            k += 1

            # Broadcast compatibility threshold and stagnation limit at the beginning
            compatibility_threshold = comm.bcast(self.config.species_set_config.compatibility_threshold if rank == 0 else None, root=0)
            self.config.species_set_config.compatibility_threshold = compatibility_threshold

            stagnation_limit = comm.bcast(self.config.stagnation_config.max_stagnation if rank == 0 else None, root=0)
            self.config.stagnation_config.max_stagnation = stagnation_limit

            comm.Barrier()  # Ensure all ranks have the same values before proceeding


            self.reporters.start_generation(self.generation)

            # Check for dynamic stagnation adjustment
            if rank == 0:  # Only the main rank handles this adjustment
                if self.generation in [11, 16]:
                    current_stagnation = self.config.stagnation_config.max_stagnation
                    if current_stagnation == 10:
                        self.config.stagnation_config.max_stagnation -= 2
                    elif current_stagnation == 8:
                        self.config.stagnation_config.max_stagnation -= 2
                    elif current_stagnation <= 6:
                        self.config.stagnation_config.max_stagnation -= 1

                    print(f"Generation {self.generation}: Adjusted stagnation to {self.config.stagnation_config.max_stagnation}")

            # Broadcast the updated stagnation limit and compatibility threshold
            #stagnation_limit = comm.bcast(self.config.stagnation_config.max_stagnation if rank == 0 else 0.0, root=0)
            #self.config.stagnation_config.max_stagnation = stagnation_limit

            comm.Barrier()
            
            # All ranks execute the fitness function
            should_stop = fitness_function(list(self.population.items()), self.config, k)


            if should_stop:
                if rank == 0:
                    print("Stopping evolution: Fitness function requested early stopping.")
                break


            # Rank 0 handles reporting and statistics
            if rank == 0:
                # Gather and report statistics.
                best = None
                generation_data = {
                    "generation": self.generation,
                    "population_size": len(self.population),
                    "species_count": len(self.species.species),
                    "species_details": [],
                    "best_genome": None, 
                    "compatibility_thres": self.config.species_set_config.compatibility_threshold,
                    "stagnation_limit": self.config.stagnation_config.max_stagnation
                }

                for g in self.population.values():
                    if g.fitness is None:
                        raise RuntimeError(f"Fitness not assigned to genome {g.key}")

                    if best is None: 
                        best = g
                    
                    elif g.fitness > best.fitness:
                        best = g

                # Collect species details
                for sid, species in self.species.species.items():
                    print("species details")
                    # Calculate average ethical score for the species
                    genome_ids = list(species.members.keys())
                    ethical_scores = []

                    for g_id in genome_ids:
                        if g_id in self.population:
                            ethical_score_history = self.population[g_id].ethical_score_history
                            # Extract only the scores from (idx, score) tuples
                            scores = [score for _, score in ethical_score_history]
                            if scores:  # Only calculate the average if there are scores
                                ethical_scores.append(sum(scores) / len(scores))  # Average of this genome's ethical scores

                    # Calculate the average ethical score for the species
                    avg_ethical_score = sum(ethical_scores) / len(ethical_scores) if ethical_scores else None

                    species_data = {
                        "species_id": sid,
                        "age": self.generation - species.created,
                        "size": len(species.members),
                        "fitness": species.fitness,
                        "adjusted_fitness": species.adjusted_fitness,
                        "stagnation": self.generation - species.last_improved,
                        "genome_ids": genome_ids,  # Add genome IDs in this species
                        "avg_ethical_score": avg_ethical_score  # Add aggregated ethical score
                    }
                    generation_data["species_details"].append(species_data)


                # Add the best genome of this generation
                if best is not None:
                    generation_data["best_genome"] = {
                        "key": best.key,
                        "fitness": best.fitness,
                        "size": best.size()
                    }

                # Append generation data
                evolution_data["generations"].append(generation_data)


                # Reporters handle the post-evaluation
                self.reporters.post_evaluate(self.config, self.population, self.species, best)

                # Track the best genome ever seen.
                if self.best_genome is None:
                    self.best_genome = best
                elif best.fitness > self.best_genome.fitness:
                    self.best_genome = best


                if not self.config.no_fitness_termination:
                    # End if the fitness threshold is reached.
                    fv = self.fitness_criterion(g.fitness for g in self.population.values())
                    if fv >= self.config.fitness_threshold:
                        self.reporters.found_solution(self.config, self.generation, best)
                        break


                # Create the next generation from the current generation.
                
                self.population = self.reproduction.reproduce(
                    self.config, self.species, self.config.pop_size, self.generation
                )

                # Check for complete extinction.
                if not self.species.species:
                    self.reporters.complete_extinction()

                    # If requested by the user, create a completely new population,
                    # otherwise raise an exception.
                    if self.config.reset_on_extinction:
                        self.population = self.reproduction.create_new(
                            self.config.genome_type,
                            self.config.genome_config,
                            self.config.pop_size
                        )
                    else:
                        raise CompleteExtinctionException()

                # Divide the new population into species.
                self.species.speciate(self.config, self.population, self.generation)

                if rank == 0:
                    species_count = len(self.species.species)
                    desired_min_species = 8
                    desired_max_species = 24
                    self.adjust_compatibility_threshold(
                        self.config,
                        species_count,
                        desired_min_species,
                        desired_max_species,
                        adjustment_step=0.5,
                    )


                if rank == 0:
                    self.reporters.end_generation(self.config, self.population, self.species)

                self.generation += 1

        # At the end of evolution, rank 0 saves the results
        if rank == 0:
            if self.config.no_fitness_termination:
                self.reporters.found_solution(self.config, self.generation, self.best_genome)

            # Save the evolution data to a JSON file
            with open(f"121_prod_evolution_generation_data_{neat_iteration}.json", "w") as json_file:
                json.dump(evolution_data, json_file, indent=4)

        # Broadcast the best genome to all ranks
        serialized_best_genome = comm.bcast(pickle.dumps(self.best_genome) if rank == 0 else None, root=0)
        self.best_genome = pickle.loads(serialized_best_genome)

        return self.best_genome

