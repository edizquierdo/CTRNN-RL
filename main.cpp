#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// EA params
const int POPSIZE = 64;
const int GENS = 20;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.5;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// NS params
const int N = 2;
const double WR = 16.0;
const double BR = 16.0;
const double TMIN = 1.0;
const double TMAX = 1.0;

// Euler params
const double StepSize = 0.01;

// RL params
const double CURRENT_PERF_WINDOW = 20;  // In units of time
const int CURRENT_PERF_WIN_STEPS = int(CURRENT_PERF_WINDOW/StepSize);
const double PAST_PERF_WINDOW = 100;  	// In units of time
const int PAST_PERF_WIN_STEPS = int(PAST_PERF_WINDOW/StepSize);

// Osc task params
const double EvolTransientDuration = 20;
const double EvolEvalDuration = 60;
const int Repetitions = 10;
const double LearnTransientDuration = CURRENT_PERF_WINDOW + PAST_PERF_WINDOW; // Has to be more than Current and Past Perf Windows
const double LearnEvalDuration = 480;

const int RunDurSteps = int((LearnTransientDuration + LearnEvalDuration)/StepSize);

const int MIN_PERIOD_STEPS = int(40/StepSize); // in time steps
const int MAX_PERIOD_STEPS = int(80/StepSize); // in time steps
const double AMP_GAIN = 1.0;
const double LEARN_RATE = 0.05;
const int LearnReps = 100;

int	VectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
}

// ------------------------------------
// Oscillation Task
// ------------------------------------
double OscFitnessFunction(TVector<double> &genotype, RandomState &rs)
{
		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		double fit = 0.0;

		// For each circuit, repeat the experiment 10 times
		for (int r = 1; r <= Repetitions; r += 1) {

			// Create the agent
			CTRNN Circuit;

			// Instantiate the nervous system
			Circuit.SetCircuitSize(N,WR,BR,MIN_PERIOD_STEPS,MAX_PERIOD_STEPS);
			int k = 1;
			// Time-constants
			for (int i = 1; i <= N; i++) {
				Circuit.SetNeuronTimeConstant(i,phenotype(k));
				k++;
			}
			// Bias
			for (int i = 1; i <= N; i++) {
				Circuit.SetNeuronBias(i,phenotype(k));
				k++;
			}
			// Weights
			for (int i = 1; i <= N; i++) {
					for (int j = 1; j <= N; j++) {
						Circuit.SetConnectionWeight(i,j,phenotype(k));
						k++;
					}
			}

			// Initialize the state between [-16,16] at random
			Circuit.RandomizeCircuitState(-16.0, 16.0, rs);

			// Run the circuit for the initial transient
			for (double time = StepSize; time <= EvolTransientDuration; time += StepSize) {
					Circuit.EulerStep(StepSize);
			}

			// Run the circuit to calculate whether it's oscillating or not
			TVector<double> pastNeuronOutput(1,N);
			double activity = 0.0;
			int steps = 0;
			for (double time = StepSize; time <= EvolEvalDuration; time += StepSize) {
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Circuit.NeuronOutput(i);
					}
					Circuit.EulerStep(StepSize);
					double magnitude = 0.0;
					for (int i = 1; i <= N; i += 1) {
						magnitude += pow((Circuit.NeuronOutput(i) - pastNeuronOutput[i])/StepSize, 2);
					}
					activity += sqrt(magnitude);
					steps += 1;
			}
			fit += activity / steps;
		}
		return (fit / Repetitions) / sqrt(N);
}

double OscLearn(TVector<double> &genotype, RandomState &rs, int printmode)
{
		ofstream file;
		if (printmode == 1) {file.open("osc_traces_learning.dat");}
		if (printmode == 2) {file.open("osc_perf_learning.dat");}

		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		// For learning
		TVector<double> pastNeuronOutput(1,N);
		TVector<double> differenceHist(1,RunDurSteps);

		double fit = 0.0;
		double pastPerf = 0.0, currentPerf = 0.0;

		// For each circuit, repeat the experiment 10 times
		for (int r = 1; r <= Repetitions; r += 1) {

			// Create the agent
			CTRNN Circuit;

			// Instantiate the nervous system
			Circuit.SetCircuitSize(N,WR,BR,MIN_PERIOD_STEPS,MAX_PERIOD_STEPS);
			int k = 1;
			// Time-constants
			for (int i = 1; i <= N; i++) {
				Circuit.SetNeuronTimeConstant(i,phenotype(k));
				k++;
			}
			// Bias
			for (int i = 1; i <= N; i++) {
				Circuit.SetNeuronBias(i,phenotype(k));
				k++;
			}
			// Weights
			for (int i = 1; i <= N; i++) {
					for (int j = 1; j <= N; j++) {
						Circuit.SetConnectionWeight(i,j,phenotype(k));
						k++;
					}
			}

			// Set learning Params
			Circuit.SetAmp(AMP_GAIN);
			Circuit.SetLearnRate(LEARN_RATE);
			Circuit.SetPeriod(MIN_PERIOD_STEPS, MAX_PERIOD_STEPS);

			// Initialize the state between [-16,16] at random
			Circuit.RandomizeCircuitState(-16.0, 16.0, rs);

			// Run the circuit for the initial transient
			int step = 1;
			double magnitude = 0.0;

			for (double time = StepSize; time <= LearnTransientDuration; time += StepSize) {

					// Saving traces to file
					if (printmode == 1){
						if (int(time/StepSize) % 10 == 0)
						{
							file << time << " " << currentPerf << " " << pastPerf << " " << Circuit.amp << " ";
							for (int x = 1; x <= N; x++) {
								for (int y = 1; y <= N; y++) {
									file << Circuit.weights[x][y] << " ";
								}
							}
							file << Circuit.biases << " " << Circuit.outputs << " ";
							for (int x = 1; x <= N; x++) {
								for (int y = 1; y <= N; y++) {
									file << Circuit.weightcenters[x][y] << " ";
								}
							}
							file << endl;
						}
				 }

					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Circuit.NeuronOutput(i);
					}
					Circuit.EulerStep(StepSize);
					magnitude = 0.0;
					for (int i = 1; i <= N; i += 1) {
						magnitude += pow((Circuit.NeuronOutput(i) - pastNeuronOutput[i])/StepSize, 2);
					}
					differenceHist[step] = sqrt(magnitude);
					step++;
			}

			// Calculate time-averaged CURRENT performance
			currentPerf = 0.0;
			for (int j = 0; j < CURRENT_PERF_WIN_STEPS; j++) {
					currentPerf += differenceHist[step - j];
			}
			currentPerf = (currentPerf / CURRENT_PERF_WIN_STEPS) / sqrt(N);

			// Calculate time-averaged PAST performance
			pastPerf = 0.0;
			for (int j = CURRENT_PERF_WIN_STEPS; j < CURRENT_PERF_WIN_STEPS + PAST_PERF_WIN_STEPS; j++) {
					pastPerf += differenceHist[step - j];
			}
			pastPerf = (pastPerf / PAST_PERF_WIN_STEPS) / sqrt(N);

			// Run the circuit to calculate whether it's oscillating or not
			for (double time = StepSize; time <= LearnEvalDuration; time += StepSize) {

					// Saving traces to file
					if (printmode == 1){
						if (int(time/StepSize) % 10 == 0)
						{
							file << LearnTransientDuration + time << " " << currentPerf << " " << pastPerf << " " << Circuit.amp << " ";
							for (int x = 1; x <= N; x++) {
								for (int y = 1; y <= N; y++) {
									file << Circuit.weights[x][y] << " ";
								}
							}
							file << Circuit.biases << " " << Circuit.outputs << " ";
							for (int x = 1; x <= N; x++) {
								for (int y = 1; y <= N; y++) {
									file << Circuit.weightcenters[x][y] << " ";
								}
							}
							file << endl;
						}
				}
					if ((printmode == 2) and (int(time/StepSize) % 100 == 0)) {file << currentPerf << " ";}

					// Step system and calculate difference in output space
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Circuit.NeuronOutput(i);
					}
					Circuit.Flux(currentPerf,rs);
					Circuit.Learn(currentPerf-pastPerf);
					Circuit.EulerStep(StepSize);
					magnitude = 0.0;
					for (int i = 1; i <= N; i += 1) {
						magnitude += pow((Circuit.NeuronOutput(i) - pastNeuronOutput[i])/StepSize, 2);
					}
					differenceHist[step] = sqrt(magnitude);
					step++;

					// Calculate time-averaged CURRENT performance
					currentPerf = 0.0;
					for (int j = 0; j < CURRENT_PERF_WIN_STEPS; j++) {
							currentPerf += differenceHist[step - j];
					}
					currentPerf = (currentPerf / CURRENT_PERF_WIN_STEPS) / sqrt(N);

					// Calculate time-averaged PAST performance
					pastPerf = 0.0;
					for (int j = CURRENT_PERF_WIN_STEPS; j < CURRENT_PERF_WIN_STEPS + PAST_PERF_WIN_STEPS; j++) {
							pastPerf += differenceHist[step - j];
					}
					pastPerf = (pastPerf / PAST_PERF_WIN_STEPS) / sqrt(N);

			}
			fit += currentPerf;
			if (printmode == 2) {file << endl;}

		}
		if (printmode > 0){file.close();}
		return (fit / Repetitions);
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{

	TVector<double> bestVector;

	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	RandomState rs;
	// cout << TracesWi(bestVector,rs)  << endl;

	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();

	// Also show the best individual in the Circuit Model form
	BestIndividualFile.open("best.ns.dat");
	GenPhenMapping(bestVector, phenotype);
	CTRNN circuit;
	// Instantiate the nervous system
	circuit.SetCircuitSize(N,WR,BR,MIN_PERIOD_STEPS,MAX_PERIOD_STEPS);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		circuit.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		circuit.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			circuit.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	BestIndividualFile << circuit;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) {

	long randomseed = static_cast<long>(time(NULL));
	if (argc == 2)
	randomseed += atoi(argv[1]);
	TSearch s(VectSize);

	#ifdef PRINTOFILE
	ofstream file;
	file.open("evol.dat");
	cout.rdbuf(file.rdbuf());
	#endif

	// Configure the search
	s.SetRandomSeed(randomseed);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE);
	s.SetMaxGenerations(GENS);
	s.SetCrossoverProbability(CROSSPROB);
	s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
	s.SetMaxExpectedOffspring(EXPECTED);
	s.SetElitistFraction(ELITISM);
	s.SetSearchConstraint(1);
	s.SetReEvaluationFlag(1);

	// Run Stage 1
	s.SetEvaluationFunction(OscFitnessFunction);
	s.ExecuteSearch();

	ifstream genefile;
	genefile.open("best.gen.dat");
	TVector<double> genotype(1, VectSize);
	genefile >> genotype;
	RandomState rs;
	rs.SetRandomSeed(randomseed);
	double baseline;
	baseline = OscLearn(genotype,rs,2);
	genefile.close();
	return 0;
}
