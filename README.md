# masterthesis-code
## Code submitted with my master thesis.


The implementation for the results shown in this thesis were done in the Julia programming language. The simple SIR model was encoded using the modeling packages of the organization *SciML* [https://docs.sciml.ai/Overview/stable/]. The virus-variant model was encoded using SBML and imported to Julia with the `SBMLToolkit` library *SBMLToolkit* [https://github.com/SciML/SBMLToolkit.jl]. Further creation of jump process, ODE and SDE representations was achieved by the corresponding packages in Julia's SciML organization. The positive Euler scheme for solving the SDEs and the Particle Filters were taken from the implementation of Lorenzo Contentos repository *Particles* [https://github.com/lcontento/Particles.jl] in the version from the 03.02.2023 after commit 01c3250. All used Julia packages and the dependencies are specified in the `Project.toml` and the `Manifest.toml` files. As an outer MCMC scheme we used the adaptive MH algorithm from the Python package `PyPESTO`. We used Python 3.9 and Julia 1.8.2 for all the computations. The sampling was performed on the high-performance computing cluster *Unicorn* of the research group of Prof. Jan Hasenauer with AMD EPYC 7F72 3.20 GHz CPUs.
The complete code can be found on the attached CD and after installing all the dependencies the code can run without further modifications. For more code and results, please contact the author of the thesis. The attached code is organised as follows.

### SIR model
This folder contains all the code to perform inference for the basic SIR model. 
- The `data` folder including the synthetically generated data.
- The `output` folder including the samples from the PMMH runs, the corresponding runtimes and the exploration of the likelihood landscapes for both parameters
- The introduction notebook `SIR-model-intro.ipynb` which shortly introduces the model definition in the three different mathematical representations. Furthermore, the synthetic data is generated and the functions for the tuning of the particle number are showcased. 
- The notebook `SIR-model-likelihoods.ipynb` contains the visualization and analysis of the likelihood landscapes for the two parameters.
 - `SIR-model-inference.ipynb` contains the definition of the full PMMH algorithm with the adaptive MH algorithm from PyPESTO. Moreover, the results for 60, 80 and 100 particles are evaluated.
- `sir_model.jl` includes the code to create the model, observation function, loading the data and the definition of the likelihood function.
- `likelihoods.jl` includes the code for the explorative analysis of the likelihood landscape.
 - `parallel_sampling.jl` was used to perform the full PMHH run. We ran 4 chains in parallel using Julia's distributed computation utility.
- All self-written functions for evaluating and plotting the results and the particle number tuning are stored in `utilities.jl`.

### Virus-variant model
This folder contains all the code to perform inference for the virus-variant model.
- The `data` folder with the synthetically generated data.
- The `petab_models` folder with the PEtab model from the supplementary code of \cite{ethiopian_data}. This includes the model specification in `covid_ethiopia_seir_variant_model_transformed_real_updated2.sbml` and the data in the `measurements.tsv` file.
- The `output` folder contains the result files that were used for the evaluations in the last chapter of this thesis.
- The model definition using the synthetically generated data can be found in `real_model_synth_data.jl`
 - In `real_model_real_data.jl` is the definition of the model using the real data.
- Likelihoods landscapes were explored with the code in `likelihood_test.jl`.
- The run of four instances of the PMMH algorithm is is implemented in `parallel_sampling.jl`for the synthetic data model and in `parallel_sampling_with_scaling.jl` for the real data model.
- Functions for the evaluation of the results and the tuning of the particle number are stored in `utilities.jl`
- An introduction to the model with all three mathematical representations is given in `intro-virus-variant-model.ipynb
- The results for the synthetic data model are evaluated in synthetic_results.ipynb`.
- A short evaluation of the results of the real data model is in `real_results.ipyb`.

