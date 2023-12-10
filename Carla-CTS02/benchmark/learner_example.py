from benchmark.rlagent import RLAgent


class Learner(RLAgent):
    def __init__(self, world, carla_map, scenario, eval_mode=False):
        super(Learner, self).__init__(world, carla_map, scenario)

    def run_step(self, debug=False):
        return super(Learner, self).run_step()
