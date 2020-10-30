import unittest
import numpy as np
import train_env as bind


startState1 = bind.State(1, 2)
startState2 = bind.State(0, 2)
startStates = list([startState1, startState2])
goalState1 = bind.State(2, 1)
goalState2 = bind.State(2, 0)
goalStates = list([goalState1, goalState2])
ob1 = bind.State(1, 1)
ob2 = bind.State(2, 2)
obSet = set([ob1, ob2])


class TestPybind(unittest.TestCase):
    def testTrainEnv(self):
        trainEnv = bind.TrainEnv(3, 3, obSet, startStates, goalStates)
        actions = list([bind.Wait, bind.Down])
        """[summary]
            Update function for train environment.
        Args:
            actions {list[Action]}  -- input list of actions for each agent
        Returns:
            States {list[TrainState]} --
                TrainState.obs_map {int[9][9]} -- 
                    obstacle observation map around each agent, agent position in center, 1 for obstacle, 0 for non-obstacle
                TrainState.pos_map {int[9][9]} -- 
                    position map around each agent, current agent position in center, non-zero cell means there is an agent with id
                TrainState.goal_map {int[9][9]} -- 
                    cost map to goal around each agent, current agent position in center, non-zero cell denotes the heuristic function to goal provide by A*, INT_MAX denotes obstacle cell or cell as other agent's goal
                TrainState.form_map {int[9][9]} -- 
                    formation map around each agent, current agent position in center, non-zero cell means there is goal of an agent with id
                TrainState.goal_vector {tuple(double, double, double)} --
                    dx,dy,magnitude of goal_vector of current agent
            reward {tuple(valid, isGoal, floss)} 
                valid {bool} -- Whether this step move is valid (not collides with other agents and obstacles)
                isGoal {list[bool]} -- If each agent reaches its goal
                floss {float} -- Formation diviation after this move 
        """
        (States, (valid, isGoal, floss)) = trainEnv.update(
            list([bind.Down, bind.Wait]))
        self.assertFalse(valid)
        trainEnv.reset()
        (States, (valid, isGoal, floss)) = trainEnv.update(
            list([bind.Left, bind.Wait]))
        self.assertFalse(valid)
        trainEnv.reset()
        (States, (valid, isGoal, floss)) = trainEnv.update(
            list([bind.Left, bind.Right]))
        self.assertFalse(valid)
        trainEnv.reset()
        (States, (valid, isGoal, floss)) = trainEnv.update(actions)
        (o_map, p_map, g_map) = trainEnv.getGlobalState()
        self.assertTrue(valid)
        self.assertEqual(isGoal, list([False, False]))
        self.assertEqual(floss, 0.2928932188134525)
        self.assertEqual(States[1].goal_vector, tuple(
            [0.8944271909999159, -0.4472135954999579, 2.23606797749979]))
        self.assertEqual(trainEnv.isSolution(goalStates), list([True, True]))
        self.assertEqual(o_map, [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(p_map, [[0, 2, 0], [0, 0, 1], [0, 0, 0]])

        trainEnv = bind.TrainEnv("src/test.yaml")
        self.assertEqual(trainEnv.getParam(), tuple([4, 20, 20]))

    def testPlanner(self):
        # AStar planner test
        plannerEnv = bind.AstarEnvironment(3, 3, obSet, goalStates[0])
        planner = bind.AStar(plannerEnv)
        solution = bind.PlanResult()
        self.assertTrue(planner.search(startStates[0], solution, 1))
        # print("AStar Planning Actions:",solution.actions)
        self.assertFalse(plannerEnv.isSolution(ob1))
        self.assertTrue(plannerEnv.isSolution(goalStates[0]))
        self.assertFalse(plannerEnv.stateValid(ob1))
        self.assertTrue(plannerEnv.stateValid(goalStates[0]))


if __name__ == '__main__':
    unittest.main()
