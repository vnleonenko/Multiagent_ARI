from simulation_influenza import Main

if __name__ == '__main__':


    pool = Main(
            strains_keys  = ['H1N1', 'H3N2', 'B'], 
            infected_init = [10, 0, 0], 
            alpha         = [0.78, 0.74, 0.6], 
            lmbd          = 0.4
            )

    pool.runs_params(
            num_runs = 5, 
            days = [1, 50],
            data_folder = 'synthetic_sample_4000',
            )

    pool.age_groups_params(
            age_groups = ['0-10', '11-17', '18-59', '60-150'], 
            vaccined_fraction = [0, 0, 0, 0]
            )

    pool.start()
