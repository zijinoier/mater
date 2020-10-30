#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>

#include <a_star.hpp>
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <numeric>
#include <vector>

using libMultiRobotPlanning::AStar;
using libMultiRobotPlanning::Neighbor;
using libMultiRobotPlanning::PlanResult;

#define OBSERVATION_R 9

struct State {
  State(int x, int y) : x(x), y(y) {}
  State() {}
  State(const State&) = default;
  State(State&&) = default;
  State& operator=(const State&) = default;
  State& operator=(State&&) = default;

  bool operator==(const State& other) const {
    return std::tie(x, y) == std::tie(other.x, other.y);
  }

  friend std::ostream& operator<<(std::ostream& os, const State& s) {
    return os << "(" << s.x << "," << s.y << ")";
  }

  double x;
  double y;
};

namespace std {
template <>
struct hash<State> {
  size_t operator()(const State& s) const {
    // http://boost.ez2learn.com/doc/html/hash/combine.html
    size_t seed = 0;
    boost::hash_combine(seed, s.x);
    boost::hash_combine(seed, s.y);
    return seed;
  }
};
}  // namespace std

struct TrainState {
  std::vector<std::vector<int>> obs_map = std::vector<std::vector<int>>(
                                    OBSERVATION_R,
                                    std::vector<int>(OBSERVATION_R, 0)),
                                pos_map = std::vector<std::vector<int>>(
                                    OBSERVATION_R,
                                    std::vector<int>(OBSERVATION_R, 0)),
                                form_map = std::vector<std::vector<int>>(
                                    OBSERVATION_R,
                                    std::vector<int>(OBSERVATION_R, 0)),
                                goal_map = std::vector<std::vector<int>>(
                                    OBSERVATION_R,
                                    std::vector<int>(OBSERVATION_R, 0));
  std::tuple<double, double, double> goal_vector;
};

enum class Action {
  Up,
  Down,
  Left,
  Right,
  Wait,
};

std::ostream& operator<<(std::ostream& os, const Action& a) {
  switch (a) {
    case Action::Up:
      os << "Up";
      break;
    case Action::Down:
      os << "Down";
      break;
    case Action::Left:
      os << "Left";
      break;
    case Action::Right:
      os << "Right";
      break;
    case Action::Wait:
      os << "Wait";
      break;
  }
  return os;
}

class AstarEnvironment {
 public:
  AstarEnvironment(size_t dimx, size_t dimy,
                   std::unordered_set<State> obstacles, State goal)
      : m_dimx(dimx),
        m_dimy(dimy),
        m_obstacles(std::move(obstacles)),
        m_goal(std::move(goal))  // NOLINT
  {}

  int admissibleHeuristic(const State& s) {
    return std::abs(s.x - m_goal.x) + std::abs(s.y - m_goal.y);
  }

  bool isSolution(const State& s) { return s == m_goal; }

  void getNeighbors(const State& s,
                    std::vector<Neighbor<State, Action, int>>& neighbors) {
    neighbors.clear();

    State up(s.x, s.y + 1);
    if (stateValid(up)) {
      neighbors.emplace_back(Neighbor<State, Action, int>(up, Action::Up, 1));
    }
    State down(s.x, s.y - 1);
    if (stateValid(down)) {
      neighbors.emplace_back(
          Neighbor<State, Action, int>(down, Action::Down, 1));
    }
    State left(s.x - 1, s.y);
    if (stateValid(left)) {
      neighbors.emplace_back(
          Neighbor<State, Action, int>(left, Action::Left, 1));
    }
    State right(s.x + 1, s.y);
    if (stateValid(right)) {
      neighbors.emplace_back(
          Neighbor<State, Action, int>(right, Action::Right, 1));
    }
  }

  void onExpandNode(const State& /*s*/, int /*fScore*/, int /*gScore*/) {}

  void onDiscover(const State& /*s*/, int /*fScore*/, int /*gScore*/) {}

 public:
  bool stateValid(const State& s) {
    return s.x >= 0 && s.x < m_dimx && s.y >= 0 && s.y < m_dimy &&
           m_obstacles.find(s) == m_obstacles.end();
  }

 private:
  int m_dimx;
  int m_dimy;
  std::unordered_set<State> m_obstacles;
  State m_goal;
};

class Environment {
 public:
  Environment(size_t dimx, size_t dimy, std::unordered_set<State> obstacles,
              std::vector<State> states, std::vector<State> goals)
      : m_dimx(dimx),
        m_dimy(dimy),
        m_obstacles(std::move(obstacles)),
        m_poss(states),
        m_starts(states),
        m_goals(std::move(goals)) {
    // std::cout << "Having built an environment containing "
    //   << m_obstacles.size()<< " Obstacles\t"
    //   << m_states.size()<<" Starting states\t"
    //   << m_goals.size()<<" Goal points\n";
    holonomic_cost_map = std::vector<std::vector<std::vector<float>>>(
        m_starts.size(),
        std::vector<std::vector<float>>(m_dimx, std::vector<float>(m_dimy, 0)));
    updateCostmap();
    if (m_poss.size() != m_goals.size())
      std::cout << "ERROR: start points size:" << m_poss.size()
                << " inequal of ending points size" << m_goals.size();
    // for (size_t j=dimy; j--;)
    // {
    //   for (size_t i=0;i<dimx;i++)
    //     std::cout<<map[i][j]<<" ";
    //   std::cout<<std::endl;
    // }
  }

  Environment(std::string inputFile) {
    YAML::Node config = YAML::LoadFile(inputFile);
    const auto& dim = config["map"]["dimensions"];
    m_dimx = dim[0].as<int>();
    m_dimy = dim[1].as<int>();

    for (const auto& node : config["map"]["obstacles"]) {
      m_obstacles.insert(State(node[0].as<int>(), node[1].as<int>()));
    }

    for (const auto& node : config["agents"]) {
      const auto& start = node["start"];
      const auto& goal = node["goal"];
      m_starts.emplace_back(State(start[0].as<int>(), start[1].as<int>()));
      m_poss.emplace_back(State(start[0].as<int>(), start[1].as<int>()));
      m_goals.emplace_back(State(goal[0].as<int>(), goal[1].as<int>()));
    }
    holonomic_cost_map = std::vector<std::vector<std::vector<float>>>(
        m_starts.size(),
        std::vector<std::vector<float>>(m_dimx, std::vector<float>(m_dimy, 0)));
    updateCostmap();

    std::cout << "\nHaving built an environment from file: " << inputFile
              << "\nContaining " << m_obstacles.size() << " Obstacles "
              << m_starts.size() << " Starting points " << m_goals.size()
              << " Goal points\n";
  }

  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  struct compare_node {
    bool operator()(const std::pair<State, double>& n1,
                    const std::pair<State, double>& n2) const {
      return (n1.second > n2.second);
    }
  };

  void updateCostmap() {
    boost::heap::fibonacci_heap<std::pair<State, double>,
                                boost::heap::compare<compare_node>>
        heap;
    for (size_t i = 0; i < m_goals.size(); i++) {
      heap.clear();

      int goal_x = (int)m_goals[i].x;
      int goal_y = (int)m_goals[i].y;
      heap.push(std::make_pair(State(goal_x, goal_y), 0));

      while (!heap.empty()) {
        std::pair<State, double> node = heap.top();
        heap.pop();

        int x = node.first.x;
        int y = node.first.y;
        for (int dx = -1; dx <= 1; dx++)
          for (int dy = -1; dy <= 1; dy++) {
            if (abs(dx) == abs(dy)) continue;
            int new_x = x + dx;
            int new_y = y + dy;
            if (new_x == goal_x && new_y == goal_y) continue;
            if (new_x >= 0 && new_x < (int)m_dimx && new_y >= 0 &&
                new_y < (int)m_dimy &&
                holonomic_cost_map[i][new_x][new_y] == 0 &&
                m_obstacles.find(State(new_x, new_y)) == m_obstacles.end()) {
              // FIXME: +1
              holonomic_cost_map[i][new_x][new_y] =
                  holonomic_cost_map[i][x][y] + 1;
              heap.push(std::make_pair(State(new_x, new_y),
                                       holonomic_cost_map[i][new_x][new_y]));
            }
          }
      }
      for (auto it = m_obstacles.begin(); it != m_obstacles.end(); it++) {
        holonomic_cost_map[i][it->x][it->y] = 1000;
      }
    }
    // for (size_t idx = 0; idx < m_goals.size(); idx++) {
    //   for (size_t i = 0; i < m_dimx; i++) {
    //     for (size_t j = 0; j < m_dimy; j++)
    //       std::cout << holonomic_cost_map[idx][i][j] << "\t";
    //     std::cout << std::endl;
    //   }
    //   std::cout << "----------------------\n";
    // }
  }

  auto getEnviromentParam() {
    return std::make_tuple(m_starts.size(), m_dimx, m_dimy);
  }

  std::vector<bool> isSolution(
      std::vector<State> states = std::vector<State>()) {
    if (states.empty()) {
      // Using default input
      states = m_poss;
    }
    std::vector<bool> result;
    for (auto it = states.begin(); it != states.end(); ++it) {
      if (!(*it.base() == m_goals.at(it - states.begin())))
        result.emplace_back(false);
      else
        result.emplace_back(true);
    }
    return result;
  }

  double ProcrustesDistance(std::vector<State> states1,
                            std::vector<State> states2) {
    // https://en.wikipedia.org/wiki/Procrustes_analysis
    // Translation
    // std::cout<<states1[0]<<" "<<states1[1]<<" "<<states2[0]<<"
    // "<<states2[1]<<std::endl;
    auto acc_x = [](int sum, State s) { return sum + s.x; };
    auto acc_y = [](int sum, State s) { return sum + s.y; };
    double mean1_x =
        std::accumulate(states1.begin(), states1.end(), 0.0, acc_x) /
        states1.size();
    double mean1_y =
        std::accumulate(states1.begin(), states1.end(), 0.0, acc_y) /
        states1.size();
    std::for_each(states1.begin(), states1.end(),
                  [&mean1_x, &mean1_y](State& s) {
                    s.x -= mean1_x;
                    s.y -= mean1_y;
                  });
    double mean2_x =
        std::accumulate(states2.begin(), states2.end(), 0.0, acc_x) /
        states2.size();
    double mean2_y =
        std::accumulate(states2.begin(), states2.end(), 0.0, acc_y) /
        states2.size();
    std::for_each(states2.begin(), states2.end(),
                  [&mean2_x, &mean2_y](State& s) {
                    s.x -= mean2_x;
                    s.y -= mean2_y;
                  });
    // Uniform scaling is NOT USED here
    // Rotation
    // std::cout<<states1[0]<<" "<<states1[1]<<" "<<states2[0]<<"
    // "<<states2[1]<<std::endl;
    double theta, numerator = 0.0, denominator = 0.0;
    for (size_t i = 0; i < states1.size(); i++) {
      numerator += (states2[i].x * states1[i].y - states2[i].y * states1[i].x);
      denominator +=
          (states2[i].x * states1[i].x + states2[i].y * states1[i].y);
    }
    theta = atan2(numerator, denominator);
    for (size_t i = 0; i < states2.size(); i++) {
      State temps;
      temps.x = cos(theta) * states2[i].x - sin(theta) * states2[i].y;
      temps.y = sin(theta) * states2[i].x + cos(theta) * states2[i].y;
      states2[i] = temps;
    }
    // std::cout<<states1[0]<<" "<<states1[1]<<" "<<states2[0]<<"
    // "<<states2[1]<<std::endl; Shape comparison
    double d = 0;
    for (size_t i = 0; i < states1.size(); i++) {
      // std::cout<<"d"<<d;
      d += pow((states1[i].x - states2[i].x), 2) +
           pow((states1[i].y - states2[i].y), 2);
    }
    d = d < 1e-9 ? 0 : sqrt(d);
    //  std::cout << "Result:" << d <<std::endl;
    return d;
  }

  auto getStateRewards(bool valid) {
    double formation_loss = ProcrustesDistance(m_poss, m_goals);
    std::vector<bool> isGoal = isSolution(m_poss);
    std::tuple<bool, std::vector<bool>, double> reward =
        std::make_tuple(valid, isGoal, formation_loss);

    std::vector<TrainState> tState;
    for (auto it = m_poss.begin(); it != m_poss.end(); it++) {
      TrainState tempState;
      // obs_map
      for (size_t i = 0; i < OBSERVATION_R; i++)
        for (size_t j = 0; j < OBSERVATION_R; j++) {
          State index(it->x + i - (int)OBSERVATION_R / 2,
                      it->y + j - (int)OBSERVATION_R / 2);
          if (stateValid(index))
            tempState.obs_map[i][j] = 0;
          else
            tempState.obs_map[i][j] = 1;
        }
      // pos_map
      for (auto p_it = m_poss.begin(); p_it != m_poss.end(); p_it++) {
        int index_x = p_it->x - it->x;
        int index_y = p_it->y - it->y;
        if (abs(index_x) <= OBSERVATION_R / 2 &&
            abs(index_y) <= OBSERVATION_R / 2)
          tempState.pos_map[index_x + (int)OBSERVATION_R / 2]
                           [index_y + (int)OBSERVATION_R / 2] =
              p_it - m_poss.begin() + 1;
      }
      // cost_map
      for (size_t i = 0; i < OBSERVATION_R; i++)
        for (size_t j = 0; j < OBSERVATION_R; j++) {
          State index(it->x + i - (int)OBSERVATION_R / 2,
                      it->y + j - (int)OBSERVATION_R / 2);
          if (stateValid(index))
            tempState.goal_map[i][j] =
                holonomic_cost_map[it - m_poss.begin()][index.x][index.y];
          else
            tempState.goal_map[i][j] = 1000;
        }
      // other agent goal mark as not valid
      for (auto p_it = m_goals.begin(); p_it != m_goals.end(); p_it++) {
        if (it - m_poss.begin() != p_it - m_goals.begin()) {
          if (m_poss[p_it - m_goals.begin()] == *p_it) {
            // std::cout << " \033[33m WARNING  :An agent " << it -
            // m_poss.begin()
            //           << "has reaches it goal!\033[0m" << std::endl;
            int index_x = p_it->x - it->x;
            int index_y = p_it->y - it->y;
            if (abs(index_x) <= OBSERVATION_R / 2 &&
                abs(index_y) <= OBSERVATION_R / 2)
              tempState.goal_map[index_x + (int)OBSERVATION_R / 2]
                                [index_y + (int)OBSERVATION_R / 2] = 3000;
          }
        }
      }
      // form_map
      for (auto p_it = m_goals.begin(); p_it != m_goals.end(); p_it++) {
        int index = it - m_poss.begin();
        int index_x = p_it->x - m_goals[index].x;
        int index_y = p_it->y - m_goals[index].y;
        if (abs(index_x) <= OBSERVATION_R / 2 &&
            abs(index_y) <= OBSERVATION_R / 2)
          tempState.form_map[index_x + (int)OBSERVATION_R / 2]
                            [index_y + (int)OBSERVATION_R / 2] =
              p_it - m_goals.begin() + 1;
      }
      // goal_vector
      int index = it - m_poss.begin();
      int goal_dx = m_goals[index].x - it->x;
      int goal_dy = m_goals[index].y - it->y;
      double magnitude = sqrt(pow(goal_dx, 2) + pow(goal_dy, 2));
      if (magnitude < 1e-8)
        tempState.goal_vector = std::make_tuple(0, 0, 0);
      else
        tempState.goal_vector =
            std::make_tuple((double)goal_dx / magnitude,
                            (double)goal_dy / magnitude, magnitude);

      tState.emplace_back(tempState);
    }

    // for (auto it = tState.begin(); it != tState.end(); it++) {
    //   std::cout << "\n--------------id:" << it - tState.begin()
    //             << "------------" << std::endl;
    //   for (size_t j = OBSERVATION_R; j--;) {
    //     for (size_t i = 0; i < OBSERVATION_R; i++)
    //       std::cout << it->obs_map[i][j] << " ";
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    //   for (size_t j = OBSERVATION_R; j--;) {
    //     for (size_t i = 0; i < OBSERVATION_R; i++)
    //       std::cout << it->pos_map[i][j] << " ";
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    //   for (size_t j = OBSERVATION_R; j--;) {
    //     for (size_t i = 0; i < OBSERVATION_R; i++)
    //       std::cout << it->goal_map[i][j] << "\t";
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    //   for (size_t j = OBSERVATION_R; j--;) {
    //     for (size_t i = 0; i < OBSERVATION_R; i++)
    //       std::cout << it->form_map[i][j] << "\t";
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    //   std::cout << "goal vector:" << std::get<0>(it->goal_vector) << " "
    //             << std::get<1>(it->goal_vector) << "@"
    //             << std::get<2>(it->goal_vector) << std::endl;
    // }

    return std::make_tuple(tState, reward);
  }

  auto getGlobalState() {
    std::vector<std::vector<int>> obs_map = std::vector<std::vector<int>>(
                                      m_dimx, std::vector<int>(m_dimy, 0)),
                                  pos_map = std::vector<std::vector<int>>(
                                      m_dimx, std::vector<int>(m_dimy, 0)),
                                  goal_map = std::vector<std::vector<int>>(
                                      m_dimx, std::vector<int>(m_dimy, 0));
    for (size_t i = 0; i < m_dimx; i++)
      for (size_t j = 0; j < m_dimy; j++) {
        if (stateValid(State(i, j)))
          obs_map[i][j] = 0;
        else
          obs_map[i][j] = 1;
      }
    for (auto it = m_poss.begin(); it != m_poss.end(); it++) {
      pos_map[it->x][it->y] = it - m_poss.begin() + 1;
    }
    for (auto it = m_goals.begin(); it != m_goals.end(); it++) {
      goal_map[it->x][it->y] = it - m_goals.begin() + 1;
    }

    // std::cout << "-------------------GLOBAL---------------------\n";
    // for (size_t j = m_dimy; j--;) {
    //   for (size_t i = 0; i < m_dimx; i++) std::cout << obs_map[i][j] << "
    //   "; std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for (size_t j = m_dimy; j--;) {
    //   for (size_t i = 0; i < m_dimx; i++) std::cout << pos_map[i][j] << "
    //   "; std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for (size_t j = m_dimy; j--;) {
    //   for (size_t i = 0; i < m_dimx; i++) std::cout << goal_map[i][j] << "
    //   "; std::cout << std::endl;
    // }
    // std::cout << std::endl;
    return std::make_tuple(obs_map, pos_map, goal_map);
  }

  auto reset() {
    m_poss.clear();
    m_poss = m_starts;
    bool valid = true;
    return getStateRewards(valid);
  }

  /**
   * update states via actions
   * actions: input actions of this time stamp
   * valid: if this update valid (such as collide with obstacles and other
   *agents) nextStates: states after this update reward: reward of this action
   **/
  auto update(const std::vector<Action> actions) {
    if (actions.size() != m_poss.size())
      std::cout << "\033[1m\033[31m"
                << "WARNING: The input action size NOT equal with agent "
                   "size!!\033[0m\n";
    bool valid = true;
    std::vector<State> nextStates;
    nextStates.clear();
    for (auto it = actions.begin(); it != actions.end(); it++) {
      State tempState;
      int index = it - actions.begin();
      switch (*it.base()) {
        case Action::Up:
          tempState = State(m_poss[index].x, m_poss[index].y + 1);
          break;
        case Action::Down:
          tempState = State(m_poss[index].x, m_poss[index].y - 1);
          break;
        case Action::Left:
          tempState = State(m_poss[index].x - 1, m_poss[index].y);
          break;
        case Action::Right:
          tempState = State(m_poss[index].x + 1, m_poss[index].y);
          break;
        case Action::Wait:
          tempState = State(m_poss[index].x, m_poss[index].y);
          break;
        default:
          std::cout << "Warning: actions has unrecognize actions type!\n";
          break;
      }
      if (std::find(nextStates.begin(), nextStates.end(), tempState) !=
              nextStates.end()
          // vertex conflict
          || !stateValid(tempState)
          // obstacle conflict
      ) {
        valid = false;
      }
      // edge conflict
      auto find_it =
          std::find(nextStates.begin(), nextStates.end(), m_poss[index]);
      if (find_it != nextStates.end()) {
        int i = std::distance(nextStates.begin(), find_it);
        if (tempState == m_poss[i]) valid = false;
      }
      nextStates.emplace_back(tempState);
    }  // end for

    m_poss.clear();
    m_poss = nextStates;

    return getStateRewards(valid);
  }

 private:
  bool stateValid(const State& s) {
    return s.x >= 0 && s.x < m_dimx && s.y >= 0 && s.y < m_dimy &&
           m_obstacles.find(s) == m_obstacles.end();
  }

 private:
  size_t m_dimx;
  size_t m_dimy;
  std::unordered_set<State> m_obstacles;
  std::vector<State> m_poss;
  std::vector<State> m_starts;
  std::vector<State> m_goals;
  std::vector<std::vector<std::vector<float>>> holonomic_cost_map;
  // std::vector< std::vector<int> > m_heuristic;
};

PYBIND11_MODULE(train_env, m) {
  m.doc() =
      "An environment for Reinforcement Learning to train multi-agent "
      "formation problem.";
  namespace py = pybind11;

  py::class_<State>(m, "State")
      .def(py::init<int, int>())
      .def_readwrite("x", &State::x)
      .def_readwrite("y", &State::y);
  py::class_<TrainState>(m, "TrainState")
      .def(py::init())
      .def_readwrite("obs_map", &TrainState::obs_map)
      .def_readwrite("pos_map", &TrainState::pos_map)
      .def_readwrite("form_map", &TrainState::form_map)
      .def_readwrite("goal_map", &TrainState::goal_map)
      .def_readwrite("goal_vector", &TrainState::goal_vector);
  py::enum_<Action>(m, "Action")
      .value("Up", Action::Up)
      .value("Down", Action::Down)
      .value("Left", Action::Left)
      .value("Right", Action::Right)
      .value("Wait", Action::Wait)
      .export_values();
  py::class_<Neighbor<State, Action, int>>(m, "Neighbor")
      .def(py::init<State, Action, int>());
  py::class_<PlanResult<State, Action, int>>(m, "PlanResult")
      .def(py::init())
      .def_readwrite("cost", &PlanResult<State, Action, int>::cost)
      .def_readwrite("fmin", &PlanResult<State, Action, int>::fmin)
      .def_readwrite("states", &PlanResult<State, Action, int>::states)
      .def_readwrite("actions", &PlanResult<State, Action, int>::actions);
  py::class_<AstarEnvironment>(m, "AstarEnvironment", "A* algrithm environment")
      .def(py::init<int, int, std::unordered_set<State>, State>(),
           "Init fuction of Astar envirnment")
      .def("admissibleHeuristic", &AstarEnvironment::admissibleHeuristic)
      .def("isSolution", &AstarEnvironment::isSolution)
      .def("stateValid", &AstarEnvironment::stateValid);
  py::class_<AStar<State, Action, int, AstarEnvironment>>(m, "AStar")
      .def(py::init<AstarEnvironment&>())
      .def("search", &AStar<State, Action, int, AstarEnvironment>::search);
  py::class_<Environment>(m, "TrainEnv")
      .def(py::init<int, int, std::unordered_set<State>, std::vector<State>,
                    std::vector<State>>())
      .def(py::init<std::string>())
      .def("getParam", &Environment::getEnviromentParam)
      .def("getGlobalState", &Environment::getGlobalState)
      .def("update", &Environment::update)
      .def("reset", &Environment::reset)
      .def("isSolution", &Environment::isSolution,
           py::arg("states") = std::vector<State>());
}