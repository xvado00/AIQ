

AIQ README
==========


For the theory behind this see:

An Approximation of the Universal Intelligence Measure
by Shane Legg and Joel Veness, 2011

and for the theory behind that see:

Universal Intelligence: A formal definition of machine intelligence
by Shane Legg and Marcus Hutter, 2007

For the theory behind the improvements see:

Towards General Evaluation of Intelligent Systems:
Lessons Learned from Reproducing AIQ Test Results
by Ondřej Vadinský, 2018

and

Towards General Evaluation of Intelligent Systems:
Using Semantic Analysis to Improve Environments in the AIQ Test
by Ondřej Vadinský, 2018

The code is released under the GNU GPLv3.  See Licence.txt file.


Known issues
------------

Sometimes a stage in the adaptive sampler completes with a test missing,
or even with an additional test run, according to the log files.  It 
seems to happen about once every 10,000 trials, based on previous
experience.  It might be a numerical bug in the way in which the true
adaptive strata targets (which are floats) are converted into integers.
It might even be OS dependent.  I just spent an hour running it on a
Mac and couldn't replicate this bug again.  Anyway, it's not really going
to change anything.  Any new adaptive stratification allocation rounds
will correct for it, plus it's only something like 1 sample out of 10k
samples so any effect should be very small compared to the general
variability in the estimates.  If anybody does track it down, let me
know and I'll fix the code.

Be careful when looking at results from the ComputeFromLog.py program
if the log file in questions is still being computed.  The reason is
that the samples tends to run through the strata in order during each
stage and thus there can be some autocorrelation in the results due
to different strata having different mean values.  So either you should
really wait until the end of a stage for the results to be sensible.
In AIQ.py the results only appear at the end of stages so this isn't
an issue.

The CI estimates when the name of samples is low seems to be too low.
We have checked the equations and everything is as it should be in
the paper.  Furthermore, as the number of samples increase past 1000
or so, our empirical tests indicate that the CI's are indeed accurate.
Maybe it's a bug in the paper where this sampler is described?  We
don't know.  In any case, do wait for 1000 samples before you place
too much confidence in the estimated CI figures.



Outline of files and directories:
---------------------------------


AIQ.py 

This is the main program to use to compute an agent's AIQ value for a
given reference machine, discount rate etc.

Arguments:

-a agent_name,param1,param2,...   

-r ref_machine_name,param1,param2,...

-d discount_rate

-l episode_length (if you don't give it it will compute one for you
 with 95% of the total infinite length episode reward covered)

-s sample_size Size of sample to use.

-n cluster_node Name of the cluster node.  Used for naming output log
 files and which sample file to read (need to think about latter
 aspect, maybe change).

-t threads_to_use Default is the number of cores on the machine.
   For a multi threaded agent you might want to set it to be less.

--log  Switch on output logging

--log_agent_failures Switch on logging of possible agent failures
	and on what program.

--verbose_log_el  Switch on logging intermediate results every 1000
  interactions, needs log-el directory.

--multi_round_el=method,param1,param2,...  Switch on multi-round EL
  convergence optimization using selected method and parameters:
	(this technique increases the consumption of RAM considerably)
    Delta,Difference,EL_to_evaluate  absolute difference of AIQ
		  is less then Difference, computed every EL_to_evaluate nr.
			of iterations.
    delta,difference,EL_to_evaluate  relative difference of AIQ
		  is less then difference in percent, computed every
			 EL_to_evaluate nr. of iterations.

--simple_mc Use a simple MC sample rather than the stratified sampler.
  Useful for sanity checks and also debugging as it doesn't do any
  async stuff etc.

--continue_from_log <path_to_log_file> Load results from
	an existing log file to continue the test from last complete stage
	(can be used to increase sample size or to salvage results of
	incomplete test runs depending on the -s value provided).

--log_interactions Logs all agent <--> environment interactions to a log file.
    Each program run has it's own log file.

--log_agent Logs agent logs from agent.get_log() to a log file.

An example run of AIQ would be:

python AIQ.py -r BF -a Q_l,0.0,0.5,0.5,0.05,0,0.9  -l 1000 -s 1000

Starting from previous log:
python AIQ.py -r BF -a Q_l,0.0,0.5,0.5,0.05,0,0.9  -l 1000 -s 1000 \
	--continue_from_log /path/to/log_file

which is a BF reference machine with a 5 symbol tape (this is the
default, you can specify other values), an episode length of 1000
with no discounting, and Q lambda with parameters...

If you want to try BF with, say, a 14 symbol tape, you'll first need
to generate a program sample file for this (see below).



ComputeFromLog.py 

Give it a log file name and it will compute the AIQ as well as results
for each strata.  At the moment you can't combined logs simply because
the first line is the program sample distribution information (needed
to work out the stratified estimate of the AIQ value).

You can provide it with multiple file names, e.g. with a *, but they
are all computed individually.

The --full option reports also the strata statistics.


/log

The raw results of runs are dumped here if logging is turned on with
the -log option in AIQ.

First line of a log file is the estimated strata distribution computed
by AIQ from the sample file, followed by lines containing a time
stamp, strata number, and result.  The file name contains the Agent,
Reference machine, discount, episode length.  You can then quickly
compute the AIQ from one of these log files later on using the program
ComputeFromLog.py (see above)



/agents

This contains the code for the various agents.

Agent.py  Base class for agents

Random.py Agent that takes random actions

Constant.py Agent that returns given constant action regardless of state.

Freq.py  Slightly smarter agent that looks at reward associated with actions.

Q_l.py  Q learning with eligibility traces.

HLQ_l.py  Like Q learning but with an automatic learning rate.

MC_AIXI.py Wrapper for Monte Carlo AIXI agent.  Must have an
executable call mc-aixi in this directory in order to run.  C++ code
for MC-AIXI can be downloaded from the internet.



/refmachines

This contains the code for the reference machines.

ReferenceMachine.py  Base class for reference machines

BF.py BF based reference machine.  Take parameters for the number of
symbols (i.e. alphabet size, default is 3) and the number of cells
that the observations use (default is 1).  Actions and rewards are
still fixed at 1 tape cell.

BF_sampler.py Generates samples of BF programs, works out their
strata, and outputs these to the terminal.  You'll want to stick these
in a sample file.  You have to name the file correctly yourself to
match what AIQ expects.  The -s option tell it how many samples to
generate. The file consists of just rows of samples so you can
concatenate the output of different runs to make a combined sample
file. The -l option specifies minimal length of generated programs
(by default, shorter programs are dropped during the sampling process,
to extend the shorter programs add option --extend_shorter.)
The --improved_optimization option enables optimization of sampled
programs by removing further pointless code.
The --improved_discriminativeness option enables optimization of
sampled programs by removing more non-discriminative programs.
The --theoretical_sampler option disables all BF code optimization
and also outputs programs that are passive or overtime.


/refmachine/sample 

Directory of program samples along with there strata.  Files
are named by the reference machine (including parameters) followed by
.sample.  This saves AIQ having to generate new samples and work out
what strata they are in, and AIQ also computes the estimated true
strata probabilities based on this sample.  So make it reasonable
large.  Say 100k programs for proper tests.  Use BF_sampler.py to
generate these for the BF reference machine.



Suggested parameters:
---------------------


To generate a BF programs sample:
python BF_sampler.py -s 200000 --improved_optimization \
  --improved_discriminativeness

To run a test with an <agent> of choice:
python AIQ.py --log --verbose_log_el --log_agent_failures \
  -r BF -a <agent> -l 100000 -s 10000
