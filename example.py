from simulation_influenza import Main

if __name__ == '__main__':


    pool = Main(
            strains_keys  = ['H1N1', 'H3N2', 'B'], 
            infected_init = [10, 0, 0], 
            alpha         = [0.78, 0.74, 0.6], 
            lmbd          = 0.3
            )

    pool.runs_params(
            num_runs = 30, 
            days = [1, 250],
            data_folder = 'sampled_200k',
            )

    pool.age_groups_params(
            age_groups = ['0-10', '11-17', '18-59', '60-150'], 
            vaccined_fraction = [0, 0, 0, 0]
            )

    pool.start(with_seirb=True)
