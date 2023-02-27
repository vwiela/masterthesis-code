# masterthesis-code
## Code submitted with my master thesis.


The implementation for the results shown in this thesis were done in the Julia programming language \cite{julia}. The simple SIR model was encoded using the modeling packages of the \textit{SciML} organization \url{https://docs.sciml.ai/Overview/stable/}. The virus-variant model was encoded using SBML \cite{sbml} and imported to Julia with the \texttt{SBMLToolkit} library \url{https://github.com/SciML/SBMLToolkit.jl}. Further creation of jump process, ODE and SDE representations was achieved by the corresponding packages in Julia's \textit{SciML} organization \cite{julia_diffeq, julia_stochdiffeq}. The positive Euler scheme for solving the SDEs and the Particle Filters were taken from the implementation of Lorenzo Contento \url{https://github.com/lcontento/Particles.jl} in the version from the 03.02.2023 after commit 01c3250. All used Julia packages and the dependencies are specified in the \texttt{Project.toml} and the \texttt{Manifest.toml} files. As an outer MCMC scheme we used the adaptive MH algorithm from the Python package \textit{PyPESTO} \cite{pypesto}. We used Python 3.9 and Julia 1.8.2 for all the computations. The sampling was performed on the high-performance computing cluster \textit{Unicorn} of the research group of Prof. Jan Hasenauer with AMD EPYC 7F72 3.20 GHz CPUs.
The complete code can be found on the attached CD and after installing all the dependencies the code can run without further modifications. For more code and results, please contact the author of the thesis. The attached code is organised as follows.

### SIR model
This folder contains all the code to perform inference for the basic SIR model. 
\begin{itemize}
    \item The \texttt{data} folder including the synthetically generated data.
    \item The \texttt{output} folder including the samples from the PMMH runs, the corresponding runtimes and the exploration of the likelihood landscapes for both parameters
    \item The introduction notebook \texttt{SIR-model-intro.ipynb} which shortly introduces the model definition in the three different mathematical representations. Furthermore, the synthetic data is generated and the functions for the tuning of the particle number are showcased. 
    \item The notebook \texttt{SIR-model-likelihoods.ipynb} contains the visualization and analysis of the likelihood landscapes for the two parameters.
    \item \texttt{SIR-model-inference.ipynb} contains the definition of the full PMMH algorithm with the adaptive MH algorithm from PyPESTO. Moreover, the results for 60, 80 and 100 particles are evaluated.
    \item \texttt{sir\_model.jl} includes the code to create the model, observation function, loading the data and the definition of the likelihood function.
    \item \texttt{likelihoods.jl} includes the code for the explorative analysis of the likelihood landscape.
    \item \texttt{parallel\_sampling.jl} was used to perform the full PMHH run. We ran 4 chains in parallel using Julia's distributed computation utility.
    \item All self-written functions for evaluating and plotting the results and the particle number tuning are stored in \texttt{utilities.jl}.
\end{itemize}

### Virus-variant model
This folder contains all the code to perform inference for the virus-variant model.
\begin{itemize}
    \item The \texttt{data} folder with the synthetically generated data.
    \item The \texttt{petab\_models} folder with the PEtab model from the supplementary code of \cite{ethiopian_data}. This includes the model specification in \texttt{covid\_ethiopia\_seir\_variant\_model\_transformed\_real\_updated2.sbml} and the data in the \texttt{measurements.tsv} file.
    \item The \texttt{output} folder contains the result files that were used for the evaluations in the last chapter of this thesis.
    \item The model definition using the synthetically generated data can be found in \texttt{real\_model\_synth\_data.jl}
    \item In \texttt{real\_model\_real\_data.jl} is the definition of the model using the real data.
    \item Likelihoods landscapes were explored with the code in \texttt{likelihood\_test.jl}.
    \item The run of four instances of the PMMH algorithm is is implemented in \texttt{parallel\_sampling.jl}for the synthetic data model and in \texttt{parallel\_sampling\_with\_scaling.jl} for the real data model.
    \item Functions for the evaluation of the results and the tuning of the particle number are stored in \texttt{utilities.jl}
    \item An introduction to the model with all three mathematical representations is given in \texttt{intro-virus-variant-model.ipynb}
    \item The results for the synthetic data model are evaluated in \texttt{synthetic\_results.ipynb}.
    \item A short evaluation of the results of the real data model is in \texttt{real\_results.ipyb}.
\end{itemize}
