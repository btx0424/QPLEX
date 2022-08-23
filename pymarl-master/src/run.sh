# example
scenario_name="academy_pass_and_shoot_with_keeper"

python main.py --config=qplex --env-config=football
    with env_args.scenario_name=${scenario_name} \
    test_interval=2000 test_nepisodes=20 \
    runner=prallel batch_size_run=8

python main.py --config=qplex_parallel --env-config=mpe \
    with env_args.scenario_name=${scenario_name} \
    test_interval=2000 test_nepisodes=20
