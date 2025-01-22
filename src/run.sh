#SBATCH --nodes=1               # Number of nodes per job (adjust as needed)
#SBATCH --ntasks=1              # Number of tasks (1 Julia process per job)
#SBATCH --time=0-23:59           # Job runtime, in D-HH:MM
#SBATCH --output=%x-New2025_62_1-10-2-%j.out       # Standard output and error log (%x: job name, %j: job ID)

# Load Julia module
module load julia
julia test_mh.jl 62 62 1 10 10 2 0 1
#for i in {1..64}
#do
#    nohup julia test_mh.jl $i $i 1 10 10 8 0 1
#done
# Loop to submit 64 jobs