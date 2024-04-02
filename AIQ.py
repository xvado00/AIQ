#!/usr/bin/python
#
# This estimates that Algorithmic Intelligence Quotient (AIQ) of a
# reinforcement learning agent by testing its performance in randomly
# sampled environments with respect to some given reference machine.
#
# Copyright Shane Legg 2011
# Copyright Ondřej Vadinský 2018, 2023
# Copyright Petr Zeman 2023
# Copyright Jan Štipl 2023
# Released under the GNU GPLv3
#

import getopt
import json
import os
import sys
from contextlib import contextmanager
import fcntl
from datetime import datetime
from hashlib import sha256
from math import isnan
from multiprocessing import get_context
from multiprocessing import Pool
from time import sleep, localtime, strftime
from typing import TextIO

import numpy as np
from numpy import ones, zeros, floor, array, sqrt, log, ceil, cov

import AIQ_continue_from_log as log_loader
from refmachines import *
from agents import *


class NpEncoder(json.JSONEncoder):
    """
    Custom encoder, because float32 can't be encoded
    https://itsourcecode.com/typeerror/typeerror-object-of-type-float32-is-not-json-serializable/
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@contextmanager
def file_lock(file_descriptor: TextIO):
    """
    Contex manager for locking access to a file
    Prevents mangled logs caused by multiprocessing
    Used in logging
    :param file_descriptor:
    :return:
    """
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)


# Test an agent by performing both positive and negative reward runs in order
# to get antithetic variance reduction.
def test_agent(refm_call, a_call, episode_length, disc_rate, stratum, program, config):
    # run twice with flipped reward second time
    s, r1, ir1, f1 = _test_agent(refm_call, a_call, 1.0, episode_length,
                                  disc_rate, stratum, program, config)
    _, r2, ir2, f2 = _test_agent(refm_call, a_call, -1.0, episode_length,
                                 disc_rate, stratum, program, config)

    # log successful result to file
    if config["logging"] and not isnan(r1) and not isnan(r2):
        with open(config["log_file_name"], 'a') as log_file:
            # optionally log also if agent failed and on what program
            if config["logging_agent_failures"]:
                log_file.write(strftime("%Y_%m%d_%H:%M:%S ", localtime()) \
                               + str(s) + " " + str(r1) + " " + str(r2) \
                               + " " + str(f1) + " " + str(f2) + " " + program + "\n")
            else:
                log_file.write(strftime("%Y_%m%d_%H:%M:%S ", localtime()) \
                               + str(s) + " " + str(r1) + " " + str(r2) + "\n")

    # log successful intermediate results to files
    if config["logging_el"] and not isnan(r1) and not isnan(r2):
        for log_el_file_name in config["log_el_files"]:
            with open(log_el_file_name, 'a') as log_el_file:
                log_el_file.write(strftime("%Y_%m%d_%H:%M:%S ", localtime()) \
                                  + str(s) + " " + str(ir1.pop(0)) + " " + str(ir2.pop(0)) + "\n")

    return s, r1, r2


# Perform a single run of an agent in an environment and collect the results
def _test_agent(refm_call, agent_call, rflip, episode_length,
                disc_rate, stratum, program, config):
    # create reference machine
    refm = eval(refm_call)

    # create agent
    agent = eval(agent_call)
    agent.reset()
    if config["log_agent"]:
        agent.set_logging(True)

    disc_reward = 0.0
    discount = 1.0

    # list of intermediate results
    disc_rewards = []

    reward, observations = refm.reset(program=program)

    mrel_stop = False
    estimated_ioc = 0

    agent_failure = '-'
    interactions = []
    for i in range(1, episode_length + 1):
        # test only if not sufficiently converged
        # or if no mrel optimization used
        if not mrel_stop:
            action = agent.perceive(observations, rflip * reward)
            if config["log_interactions"]:
                obs_str = ",".join([str(obs) for obs in observations])
                interactions.append(f"{rflip * reward},{action},{obs_str}\n")
            reward, observations, steps = refm.act(action)

            # we signal failure with a NaN so as not to upset
            # the parallel map running this with an exception
            if steps == refm.max_steps:
                return stratum, float('nan'), disc_rewards, agent_failure

            disc_reward += discount * rflip * reward
            discount *= disc_rate
            estimated_ioc = i
        # if sufficiently converged expect the same score
        # for the rest of episode
        else:
            disc_reward += discount * converged_reward
            discount *= disc_rate

        if config["logging_el"]:
            if i % config["intermediate_length"] == 0:
                intermediate_reward = normalise_reward(i, disc_rate, disc_reward)
                disc_rewards.append(intermediate_reward)

        if config["multi_rounding_el"] and not mrel_stop:
            mrel_stop, converged_reward = evaluate_mrel_stopping_condition(disc_rewards, i, config)
            config["mrel_rewards"].append(disc_reward)

    # normalise and possibly discount reward
    disc_reward = normalise_reward(episode_length, disc_rate, disc_reward)

    # get information about agent failures
    if config["logging_agent_failures"]:
        agent_failure = agent.has_failed()

    time_str = datetime.now().strftime("%Y_%m%d_%H:%M:%S.%f")

    # save debug information
    if config["debugging_mrel"]:
        if mrel_stop:
            mrel_status = "converged"
        else:
            mrel_status = "finished"
        with open(config["mrel_debug_file_name"], 'w') as mrel_debug_file:
            mrel_debug_file.write(time_str + mrel_status + " " + str(disc_reward) + " " + str(estimated_ioc)
                                  + " " + program + " " + str(rflip) + "\n")
    # save internal agent logs
    if config["log_agent"]:
        with open(config["log_agent_file_path"], "a") as agent_log:
            record = {
                "time_stamp": time_str,
                "rflip": rflip,
                "stratum": stratum,
                "program": program,
                "agent_log": agent.get_logs()
            }
            with file_lock(agent_log):
                json.dump(record, agent_log, cls=NpEncoder)
                agent_log.write("\n")

    # save agent <-> env interactions
    if config["log_interactions"]:
        program_hash = sha256(program.encode('utf-8')).hexdigest()
        file_path = f'{config["log_interactions_dir_path"]}{time_str.replace(":", "_")}_{program_hash}.csv'
        with open(file_path, "w") as interactions_log:
            interactions_log.write(f'{rflip},"{program}"\n')
            interactions_log.writelines(interactions)

    # dispose of agent and reference machine
    del agent
    del refm

    return stratum, disc_reward, disc_rewards, agent_failure


# Normalise and possibly discount reward
def normalise_reward(episode_length, disc_rate, disc_reward):
    # if discounting normalise (and thus correct for missing tail)
    if disc_rate != 1.0:
        disc_reward /= ((1.0 - disc_rate ** (episode_length + 1)) / (1.0 - disc_rate))
    else:
        # otherwise just normalise by the episode length
        disc_reward /= episode_length

    return disc_reward


# Evaluate if a stopping condition for a multi-round EL convergence optimization
# is met based on which evaluation method is used.
def evaluate_mrel_stopping_condition(disc_rewards, current_iteration, config):
    mrel_stop = False

    # Call specific evaluator
    if mrel_method == "Delta":
        mrel_stop, converged_reward = _evaluate_mrel_Delta_stopping_condition(
            disc_rewards, current_iteration, config)

    if mrel_method == "delta":
        mrel_stop, converged_reward = _evaluate_mrel_delta_stopping_condition(
            disc_rewards, current_iteration, config)

    return mrel_stop, converged_reward


# Specific evaluation methods for a multi-round EL convergence optimization
# Delta: absolute difference in score at two consecutive ELs to evaluate
# is less than a specified difference
def _evaluate_mrel_Delta_stopping_condition(disc_rewards, current_iteration, config):
    mrel_stop = False
    converged_reward = None

    # check only every mrel_Delta_el iterations
    if current_iteration % mrel_Delta_el == 0:
        # verify there is enough intermediate results to work with
        if len(disc_rewards) >= 2 * mrel_Delta_el / intermediate_length:
            reward1 = disc_rewards[-1]
            reward2 = disc_rewards[-(1 + mrel_Delta_el / intermediate_length)]
            if abs(reward1 - reward2) < mrel_Delta_diff:
                mrel_stop = True
                # compute avg reward from the converged part of interaction history
                disc_reward1 = config["mrel_rewards"][-1]
                disc_reward2 = config["mrel_rewards"][-mrel_Delta_el]
                # TODO: probably does not work with discounting
                converged_reward = normalise_reward(mrel_Delta_el, 1.0,
                                                    disc_reward1 - disc_reward2)

    return mrel_stop, converged_reward


# delta: relative difference in score at two consecutive ELs to evaluate
# is less than a specified percentage
def _evaluate_mrel_delta_stopping_condition(disc_rewards, current_iteration, config):
    mrel_stop = False
    converged_reward = None

    # check only every mrel_delta_el iterations
    if current_iteration % mrel_delta_el == 0:
        # verify there is enough intermediate results to work with
        if len(disc_rewards) >= 2 * mrel_delta_el / intermediate_length:
            reward1 = disc_rewards[-1]
            reward2 = disc_rewards[-(1 + mrel_delta_el / intermediate_length)]
            if reward1 != 0.0:
                if abs(100.0 * (reward1 - reward2) / reward1) < mrel_delta_diff:
                    mrel_stop = True
                    # compute avg reward from the converged part of interaction history
                    disc_reward1 = config["mrel_rewards"][-1]
                    disc_reward2 = config["mrel_rewards"][-mrel_Delta_el]
                    # TODO: probably does not work with discounting
                    converged_reward = normalise_reward(mrel_delta_el, 1.0,
                                                        disc_reward1 - disc_reward2)

    return mrel_stop, converged_reward


# Simple MC estimator, useful for checking the more complex adaptive estimator.
def simple_mc_estimator(refm_call, agent_call, episode_length, disc_rate,
                        sample_size, config):
    print()
    result = zeros((len(config["sample_data"])))
    i = 0  # Successful runs counter
    for stratum, program in config["sample_data"]:
        _s, r1, _r2 = test_agent(refm_call, agent_call, episode_length, disc_rate, stratum, program, config)
        if isnan(r1):
            # Input program was invalid / didn't return
            continue
        i += 1
        result[i] = r1
        if i % 10 == 0 and i > 10:
            mean = result[:i].mean()
            half_ci = 1.96 * result[:i].std(ddof=1) / sqrt(i)
            print("         %6i  % 5.1f +/- % 5.1f " % (i, mean, half_ci))

        if i >= sample_size:
            break


# Adaptive stratified estimator
#
# The following parameter names are from the paper that describes the algorithm:
#
# N total number of samples taken in each step (i.e. across all strata)
# p probability of being in a stratum
def stratified_estimator(refm_call, agent_call, episode_length, disc_rate, samples,
                         sample_size, dist, threads, config):
    p = dist  # get probability of being in each stratum
    I = len(dist)  # number of strata, including passive
    A = sum(ceil(p))  # active strata

    # With N below I keep the steps small to start with to force the system
    # sample a reasonable number of each to get better variability estimates
    # before starting to adapt more.  It would be nice to have a fully online
    # version without these steps.
    N = [0, 3 * A, 6 * A, 10 * A,
         20 * A, 30 * A, 50 * A, 70 * A, 100 * A, 250 * A, 500 * A, 750 * A, 1000 * A,
         1250 * A, 1500 * A, 1750 * A, 2000 * A, 2500 * A, 3000 * A, 3500 * A, 4000 * A, 5000 * A]

    # trim to number of program samples, or requested sample size, which ever is smaller
    max_samples = min(len(config["sample_data"]), sample_size)
    for i in range(len(N)):
        if N[i] + A >= max_samples:
            N[i] = float(max_samples)
            N = N[:i + 1]
            break

    print("Sample size steps:")
    print(N)

    K = len(N)  # number of adaptive stratification stages
    Y = [[] for _ in range(I)]  # empty collection of samples divided up by stratum
    Y[0] = [0]
    s = ones((K, I))  # estimated standard deviations for each stage & strata
    n = zeros((K, I))  # for each step the size of each stratum
    est = zeros(K)  # estimated confidence intervals

    log_results = None
    # Load checkpoint data only if the file exists
    continue_from_log_path = config.get("continue_from_log_path")
    if continue_from_log_path is not None and os.path.isfile(continue_from_log_path):
        _, log_results = log_loader.load_log_file(continue_from_log_path)
        print(f"Loaded log results from {continue_from_log_path}")

        if config.get("logging_el"):
            log_loader.copy_verbose_log_el(continue_from_log_path, config["log_file_name"], N)

    for k in range(1, K):
        print()
        # compute the allocations with "method a" from the paper,
        # deducting 2A from the target which is added afterward to ensure
        # all strata get at least 2 samples
        # We also divide by 2, than multiply by 2 later on to ensure that
        # the sample size in each stratum is even (this is for antithetic variables)

        m = (p * s[k - 1]) / sum(p * s[k - 1]) * (N[k] - N[k - 1] - 2.0 * A) / 2.0

        x = zeros((len(m)))
        x[0] = floor(m[0])
        for i in range(1, I):
            x[i] = max(floor(sum(m[:i + 1]) + 0.001) - floor(sum(m[:i])), 0)

        x *= 2.0

        # make sure each non-zero probability stratum gets sampled at least twice
        M = x + 2.0 * ceil(p)

        print_log_loading_message = False
        if continue_from_log_path is not None and len(log_results) > 0:
            is_stage_complete = read_from_log(I, M, Y, log_results, config)

            if not is_stage_complete:
                # We run out of data -> return correct results that we got from previous stages
                print("Loading of log DONE")
                print(f"Log for stage {k} is incomplete")
                print(f"Starting agent on stage {k}/{K - 1}")
                run_agent(I, M, Y, agent_call, config, disc_rate, episode_length, refm_call, samples, threads)
            else:
                # Log from this stage is complete, but the results have to be processed and then we print
                # log complete message
                print_log_loading_message = True
        else:
            run_agent(I, M, Y, agent_call, config, disc_rate, episode_length, refm_call, samples, threads)

        # compute new total program sample counts for each stratum
        n[k] = n[k - 1] + M

        # compute empirical standard deviations for each stratum
        for i in range(1, I):
            if p[i] > 0.0 and n[k][i] > 2:

                YA = array(Y[i])
                sample1 = YA[:, 0]  # positive antithetic runs
                sample2 = YA[:, 1]  # negative antithetic runs

                s1 = sample1.std(ddof=1)  # 1 degree of freedom
                s2 = sample2.std(ddof=1)  # 1 degree of freedom
                covariance = cov(sample1, sample2)[0, 1]  # default is 1 df

                var = 0.25 * (s1 * s1 + s2 * s2 + 2.0 * covariance)

                # Covariance can be negative 0 -> sqrt from negative -> NaN
                #   Cov in general can be negative, but it didn't happen in testing
                #   Wrong calculation of cov results in it being not quite zero instead of zero
                #   Will fix when the problem occurs
                if abs(var) <= 1e-10:
                    var = 0.0

                s[k, i] = sqrt(var)
            else:
                s[k, i] = 1.0

        # report current estimates by strata
        for i in range(1, I):
            print(" % 3d % 4d % 5d" % (i, int(M[i]), n[k][i]), end=' ')

            if n[k][i] == 0:
                # no samples, so skip mean and half CI
                print()
            elif n[k][i] < 4:
                # don't report half CI with less than 4 program samples
                print(" % 6.1f" % (array(Y[i]).mean()))
            else:
                # do a full report
                # statistical samples is twice program samples due to antithetic vars
                print(" % 6.1f +/- % 5.1f"
                      % (array(Y[i]).mean(), 1.96 * s[k, i] / sqrt(n[k][i])))

        # compute the current estimate and 95% confidence interval
        for i in range(1, I):
            if p[i] > 0.0:
                est[k - 1] += p[i] / (n[k][i]) * array(Y[i]).sum()

        delta = 1.96 * sum(p * s[k]) / sqrt(N[k])

        # wait until after 3rd stage due to unreliable early statistics
        if k >= min(3, K - 1):
            print("\n         %6i   % 5.1f +/- % 5.1f " % (N[k], est[k - 1], delta))

        if print_log_loading_message and len(log_results) == 0:
            print("Loading of log DONE")
            print(f"Log for stage {k} is complete")
            print(f"Starting agent on stage {k + 1}/{K - 1}")


def read_from_log(I, M, Y, log_results: list[log_loader.LogResult], config: dict) -> bool:
    """
    Reads records in log_results and fills Y with fresh results.

    :param I: number of strata, including passive
    :param M:
    :param Y: collection of results divided up by stratum
    :param log_results:
    :param config:
    :return: True, if we had enough data to complete the stage
    """

    results = []
    # Load from log
    # add samples to processing pool (we skip stratum 0 which is passive)
    # We choose number of  programs here according to M[i]
    for i in range(1, I):
        for _ in range(int(M[i]) // 2):  # /2 is due to sampling each program twice

            try:
                results.append(log_results.pop(0))
            except IndexError as _:
                return False

    # collect the results, adding new jobs to the pool for any failed runs
    while results != []:
        result = results.pop(0)

        # Write the results to new log file
        with open(config["log_file_name"], 'a') as log_file:
            if config["logging_agent_failures"]:
                log_file.write(f"{result.time} {result.stratum_number} {result.reward_1} {result.reward_2} "
                               f"{result.fail1} {result.fail2} {result.program}\n")
            else:
                log_file.write(f"{result.time} {result.stratum_number} {result.reward_1} {result.reward_2}\n")

        # Run succeeded, so add the result to our results table Y
        Y[result.stratum_number].append((result.reward_1, result.reward_2))
    return True


def run_agent(I, M, Y, agent_call, config, disc_rate, episode_length, refm_call, samples, threads):
    results = []
    # Run normally
    # create parallel computation pool
    # Fix multiprocessing issues ocurring on some Linux setups
    # https://pythonspeed.com/articles/python-multiprocessing/
    if threads == 0:
        pool = get_context("spawn").Pool()  # default threads = core count
    else:
        pool = get_context("spawn").Pool(threads)
    # add samples to processing pool (we skip stratum 0 which is passive)
    # We choose number of  programs here according to M[i]
    for i in range(1, I):
        for _ in range(int(M[i]) // 2):  # /2 is due to sampling each program twice

            if len(samples[i]) == 0:
                print("Error: Run out of program samples in stratum: " + str(i))
                sys.exit()

            program = samples[i].pop(0)
            args = (refm_call, agent_call, episode_length, disc_rate, i, program, config)
            result = pool.apply_async(test_agent, args)
            results.append(result)
    # collect the results, adding new jobs to the pool for any failed runs
    while results != []:
        result = results.pop(0)

        if not result.ready():
            # put back in the results list at the end and sleep for a moment
            results.append(result)
            sleep(0.02)
        else:
            # completed, now get the results
            stratum, perf1, perf2 = result.get(100)

            if isnan(perf1) or isnan(perf2):
                # run failed so get a new sample and add to processing pool
                # print "Adding extra sample to the pool due to run failure"
                if len(samples[stratum]) == 0:
                    print("Error: Run out of program samples in stratum: "
                          + str(stratum))
                    sys.exit()

                program = samples[stratum].pop(0)
                args = (refm_call, agent_call, episode_length, disc_rate, stratum, program, config)
                result = pool.apply_async(test_agent, args)
                results.append(result)
            else:
                # run succeeded, so add the result to our results table Y
                Y[stratum].append((perf1, perf2))
    pool.close()
    pool.join()


# load the pre-sampled programs
sample_data = None


def load_samples(refm, cluster_node, simple_mc):
    global sample_data

    program_sample_filename = "./refmachines/samples/" + str(refm) \
                              + cluster_node + ".samples"

    print("Loading program samples: " + program_sample_filename)

    with open(program_sample_filename) as sample_file:
        num_strata = 0
        sample_data = []
        for line in sample_file:
            s, prog = line.split()
            stratum = int(s)
            num_strata = max(num_strata, stratum)
            sample_data.append((stratum, prog))

    num_strata += 1  # due to strata starting at 0
    num_samples = len(sample_data)

    samples = [[] for _ in range(num_strata)]
    dist = zeros(num_strata)

    for stratum, program in sample_data:
        samples[stratum].append(program)
        dist[stratum] += 1.0 / num_samples

    print("Number of program samples:" + str(num_samples))
    if not simple_mc:
        print("Number of strata:        " + str(num_strata))
        print("Strata distribution:")
        print(str(dist[1:]))
    print()
    return samples, dist


# print basic usage
def usage():
    print("python AIQ -r reference_machine[,param1[,param2[...]]] "
          + "-a agent[,param1[,agent_param2[...]]] "
          + "-d discount_rate [-s sample_size] [-l episode_length] "
          + "[-n cluster_node] [-t threads] [--log] [--log_agent_failures] "
          + "[--verbose_log_el] [--simple_mc]"
          + "[--multi_round_el=method[,param1[,param2[...]]]"
          + "[--debug_mrel]")


# main function that just sets things up and then calls the sampler
logging = False
log_file = None
logging_el = False
log_el_files = []
intermediate_length = 1000
multi_rounding_el = False
mrel_method = None
mrel_params = []
mrel_rewards = []
debugging_mrel = False
mrel_debug_file = None
logging_agent_failures = False


def main():
    global logging, log_file
    global logging_el, log_el_files, intermediate_length
    global multi_rounding_el, mrel_method, mrel_params, mrel_rewards
    global debugging_mrel, mrel_debug_file
    global logging_agent_failures

    print()
    print("AIQ version 1.0")
    print()
    # get the command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "r:d:l:a:n:s:t:",
                                   [
                                       "multi_round_el=",
                                       "help",
                                       "log",
                                       "simple_mc",
                                       "verbose_log_el",
                                       "debug_mrel",
                                       "log_agent_failures",
                                       "continue_from_log=",
                                       "log_interactions",
                                       "log_agent"
                                   ])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    agent_str = None
    refm_str = None
    disc_rate = None
    episode_length = None
    cluster_node = ""
    simple_mc = False
    sample_size = None
    agent_params = []
    refm_params = []
    threads = 0
    continue_from_log_path = None
    log_agent = False
    log_interactions = False
    # exit on no arguments
    if opts == []:
        print("No arguments were given")
        usage()
        sys.exit()

    # parse arguments
    for opt, arg in opts:
        if opt == "-a":
            args = arg.split(",")
            agent_str = args.pop(0)
            for a in args:
                agent_params.append(float(a))

        elif opt == "-r":
            args = arg.split(",")
            refm_str = args.pop(0)
            for a in args:
                refm_params.append(float(a))

        elif opt == "-d":
            disc_rate = float(arg)
        elif opt == "-l":
            episode_length = int(arg)
        elif opt == "-s":
            sample_size = int(arg)
        elif opt == "-n":
            cluster_node = "_" + arg
        elif opt == "-t":
            threads = int(arg)
        elif opt == "--log":
            logging = True
        elif opt == "--simple_mc":
            simple_mc = True
        elif opt == "--verbose_log_el":
            logging_el = True
        elif opt == "--multi_round_el":
            multi_rounding_el = True
            args = arg.split(",")
            mrel_method = args.pop(0)
            for a in args:
                mrel_params.append(float(a))

        elif opt == "--debug_mrel":
            debugging_mrel = True
        elif opt == "--log_agent_failures":
            logging_agent_failures = True
        elif opt == "--continue_from_log":
            continue_from_log_path = arg
        elif opt == "--log_agent":
            log_agent = True
        elif opt == "--log_interactions":
            log_interactions = True
        else:
            print("Unrecognised option")
            usage()
            sys.exit()

    # basic parameter checks
    if agent_str is None:
        raise NameError("missing agent")
    if refm_str is None:
        raise NameError("missing reference machine")
    if disc_rate is None:
        disc_rate = 1.0
    if agent_str == "Manual" and not simple_mc:
        raise NameError("Manual control only works with the simple mc sampler")
    if multi_rounding_el and not logging_el:
        raise NameError("multi-round EL convergence possible only with verbose EL logging")
    if multi_rounding_el and not mrel_method in ["Delta", "delta"]:
        raise NameError("unrecognized multi-round EL convergence method \
                (only 'Delta' or 'delta' implemented)")
    if debugging_mrel and not multi_rounding_el:
        raise NameError("debugging of multi-round EL convergence possible only with multi-round EL convergence enabled")

    # Load multi-EL methods parameters or set defaults
    if multi_rounding_el:
        if mrel_method == "Delta":
            global mrel_Delta_diff, mrel_Delta_el
            mrel_Delta_diff = 0.1
            mrel_Delta_el = 1000
            if len(mrel_params) > 0:
                param = mrel_params.pop(0)
                # rewards are bounded and Delta is absolute
                if 0 <= param < 100:
                    mrel_Delta_diff = param
                else:
                    raise NameError("invalid MREL Delta parameter value: \
                            minimal difference must be in [0,100).")

                if len(mrel_params) > 0:
                    param = int(mrel_params.pop(0))
                    if intermediate_length <= param <= episode_length / 2 and param % intermediate_length == 0:
                        mrel_Delta_el = param
                    else:
                        raise NameError("invalid MREL Delta parameter value: \
                                EL to evaluate must be in [1000,EL/2) and mod(0) by 1000.")
        elif mrel_method == "delta":
            global mrel_delta_diff, mrel_delta_el
            mrel_delta_diff = 0.1
            mrel_delta_el = 1000
            if len(mrel_params) > 0:
                param = mrel_params.pop(0)
                # delta is absolute value in percent, lower values are interesting
                if 0 <= param <= 100:
                    mrel_delta_diff = param
                else:
                    raise NameError("invalid MREL delta parameter value: \
                            minimal percentage must be in [0,100].")

                if len(mrel_params) > 0:
                    param = int(mrel_params.pop(0))
                    if intermediate_length <= param <= episode_length / 2 and param % intermediate_length == 0:
                        mrel_delta_el = param
                    else:
                        raise NameError("invalid MREL delta parameter value: \
                                EL to evaluate must be in [1000,EL/2) and mod(0) by 1000.")

    # compute episode_length to have 95% of the infinite total in each episode
    # or if episode_length given compute the proportion that this gives
    proportion_of_total = 0.95
    if episode_length is None:
        if disc_rate == 1.0:
            print("With a discount rate of 1.0 you must set the episode length.")
            print()
            usage()
            sys.exit()
        else:
            episode_length = int(log(1.0 - proportion_of_total) / log(disc_rate))
    else:
        proportion_of_total = 1.0 - disc_rate ** episode_length

    # construct reference machine
    refm_call = refm_str + "." + refm_str + "( "
    if len(refm_params) > 0:
        refm_call += str(refm_params.pop(0))
    for param in refm_params:
        refm_call += ", " + str(param)
    refm_call += " )"
    refm = eval(refm_call)

    # construct agent
    agent_call = agent_str + "." + agent_str + "( refm, " + str(disc_rate)
    for param in agent_params:
        agent_call += ", " + str(param)
    agent_call += " )"
    agent = eval(agent_call)

    # report settings
    print("Reference machine:       " + str(refm))
    print("RL Agent:                " + str(agent))
    print("Discount rate:           " + str(disc_rate))
    print("Episode length:          " + str(episode_length), end=' ')
    if disc_rate != 1.0:
        print(" which covers %3.1f%% of the infinite geometric total"
              % (100.0 * proportion_of_total))
    else:
        print()
    if agent == "Manual()" and not simple_mc:
        print("Error: Manual agent only works with simple_mc sampling")
        sys.exit()

    if disc_rate != 1.0 and proportion_of_total < 0.75:
        print()
        print("WARNING: The episode length is too short for this discount rate!")
        print()
    print("Sample size:             " + str(sample_size))

    # load in program samples
    samples, dist = load_samples(refm, cluster_node, simple_mc)

    if sample_size is None:
        sample_size = len(sample_data)

    # The following is a crude check as we can still run out of samples in a
    # stratum depending on how the adaptive stratification decides to sample.
    if sample_size > 2.0 * len(sample_data):
        print()
        print("Error: More samples have been requested than are available in "
              "the program sample file! (including fact that they are sampled twice)")
        sys.exit()

    # Assignment for dictionary even if not used
    log_file_name = ''
    base_log_name = str(refm) + "_" + str(disc_rate) + "_" \
                    + str(episode_length) + "_" + str(agent) + cluster_node \
                    + strftime("_%Y_%m%d_%H_%M_%S", localtime()) + ".log"
    # report logging
    if logging:
        log_dir = "./log/"
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        log_file_name =  log_dir + base_log_name
        with open(log_file_name, 'w') as log_file:
            for i in range(1, len(dist)):
                log_file.write(str(dist[i]) + " ")
            log_file.write("\n")
        print("Logging to file:         " + log_file_name)

    # set up files to log results at intermediate ELs
    if logging_el:
        if episode_length > intermediate_length:
            log_el_dir_name = str(refm) + "_" + str(disc_rate) + "_" \
                              + str(episode_length) + "_" + str(agent) + cluster_node \
                              + strftime("_%Y_%m%d_%H_%M_%S", localtime())
            if not os.path.exists("./log-el/" + log_el_dir_name):
                os.makedirs("./log-el/" + log_el_dir_name)
            for i in range(1, episode_length // intermediate_length + 1):
                log_el_file_name = "./log-el/" + log_el_dir_name + "/" + str(refm) + "_" \
                                   + str(disc_rate) + "_" + str(i * intermediate_length) + "_" \
                                   + str(agent) + cluster_node \
                                   + strftime("_%Y_%m%d_%H_%M_%S", localtime()) + ".log"

                with open(log_el_file_name, 'w') as log_el_file:
                    for j in range(1, len(dist)):
                        log_el_file.write(str(dist[j]) + " ")
                    log_el_file.write("\n")

                log_el_files.append(log_el_file_name)
            print("Verbose logging at intermediate ELs to directory: ./log-el/" + log_el_dir_name)
        else:
            print("Warning: Episode Length " + str(episode_length) + " is less than Intermediate Episode Length "
                  + str(intermediate_length) + "! Verbose logging at Intermediate Episode Lengths will be disabled.")
            logging_el = False
            if multi_rounding_el:
                raise NameError("multi-round EL convergence possible only with verbose EL logging")

    # Assignment for dictionary even if not used
    mrel_debug_file_name = ''
    # set up file to save multi-round EL convergence debug information
    if debugging_mrel:
        mrel_debug_file_name = "./debug/" + str(refm) + "_" + str(disc_rate) + "_" \
                               + str(episode_length) + "_" + str(agent) + cluster_node \
                               + strftime("_%Y_%m%d_%H_%M_%S", localtime()) + ".log"

        with open(mrel_debug_file_name, 'w') as mrel_debug_file:
            mrel_debug_file.write("# Multi-round EL convergence method: " + mrel_method + "\n")
            mrel_debug_file.write("# Multi-round EL convergence method parameters:\n")
            if mrel_method == "Delta":
                mrel_debug_file.write("#   Delta=" + str(mrel_Delta_diff) + "\n")
                mrel_debug_file.write("#   EL=" + str(mrel_Delta_el) + "\n")
            elif mrel_method == "delta":
                mrel_debug_file.write("#   delta=" + str(mrel_delta_diff) + "\n")
                mrel_debug_file.write("#   EL=" + str(mrel_delta_el) + "\n")
        print("MREL debug logging to file:         " + mrel_debug_file_name)

    log_interactions_dir_path = f"./log-interactions/{base_log_name.rstrip('.log')}/"
    if log_interactions:
        if not os.path.exists(log_interactions_dir_path):
            os.makedirs(log_interactions_dir_path)

    log_agent_file_path = "./log-agent/"
    if log_agent:
        if not os.path.exists(log_agent_file_path):
            os.mkdir(log_agent_file_path)

    config = {
        "logging": logging,
        "log_file_name": log_file_name,
        "sample_data": sample_data,
        "logging_el": logging_el,
        "log_el_files": log_el_files,
        "intermediate_length": intermediate_length,
        "multi_rounding_el": multi_rounding_el,
        "mrel_method": mrel_method,
        "mrel_params": mrel_params,
        "mrel_rewards": mrel_rewards,
        "debugging_mrel": debugging_mrel,
        "mrel_debug_file": mrel_debug_file_name,
        "logging_agent_failures": logging_agent_failures,
        "continue_from_log_path": continue_from_log_path,
        "log_agent": log_agent,
        "log_agent_file_path": log_agent_file_path + base_log_name,
        "log_interactions": log_interactions,
        "log_interactions_dir_path": log_interactions_dir_path
    }

    # run an estimation algorithm
    if simple_mc:
        simple_mc_estimator(refm_call, agent_call, episode_length, disc_rate, sample_size, config)
    else:
        # Kill agent and pass in its constructor call, this is because on Windows
        # some agents have trouble serialising which messes up the multiprocessing
        # library that Python uses.  Easier just to construct the agent inside the
        # method that gets called in parallel.
        stratified_estimator(refm_call, agent_call, episode_length, disc_rate,
                             samples, sample_size, dist, threads, config)


if __name__ == "__main__":
    main()
