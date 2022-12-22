"""The actual implementation of the linear optimization problem."""
# from itertools import chain, permutations, product
# from typing import Iterable
#
# from pyscipopt import Model, quicksum
#
#
# class RobustnessVerification:
#     """Instances of this class represent instances of the linear optimization problem
#
#     For details see chapter 3 in [Ludwig2023]_.
#     """
#
# Parameters
# ----------
# chars : CharTuple
#     the (special) characters to be assign
# poss: PosTuple
#     the positions to which we want to assign the (special) characters
# """
#
# #
# chars: Chars
# poss: PosTuple
#
# def __init__(self, chars: Chars, poss: PosTuple):
#     assert len(chars.monos) == len(poss)
#     self.chars = chars
#     self.poss = poss
#     self.char_pos_assigns: LinVars = {}
#     self.quad_char_pos_assigns: QuadVars = {}
#     self.char_pos_costs: LinCosts = {}
#     self.quad_char_pos_costs: QuadCosts = {}
#     self.model: Model = Model("Keyboard Layout Optimization")
#
# def set_up_model(self, char_pos_costs: LinCosts, quad_char_pos_costs: QuadCosts):
#     """Set up all the variables and initialize the costs for the SCIP model"""
#     for (char, pos) in self.char_pos_assigns_keys:
#         self.char_pos_assigns[char, pos] = self.model.addVar(
#             name=f"{pos}={char}", vtype="B"
#         )
#     for (char, char_2, pos, pos_2) in self.quad_char_pos_assigns_keys:
#         self.quad_char_pos_assigns[char, char_2, pos, pos_2] = self.model.addVar(
#             name=f"{pos}={char}_and_{pos_2}={char_2}", vtype="C", lb=0, ub=1
#         )
#     assert len(quad_char_pos_costs) == len(self.quad_char_pos_assigns)
#     assert len(char_pos_costs) == len(self.char_pos_assigns)
#     self.char_pos_costs = char_pos_costs
#     self.quad_char_pos_costs = quad_char_pos_costs
#
#     constr = {}
#     for char in self.chars.monos:
#         self.model.addCons(
#             quicksum(self.char_pos_assigns[char, pos] for pos in self.poss) == 1,
#             f"AllCharacterAssignedOnce({char})",
#         )
#         for (pos, pos_2) in self.pos_pairs:
#             self.model.addCons(
#                 quicksum(
#                     self.quad_char_pos_assigns[char, char_2, pos, pos_2]
#                     for char_2 in self.chars.monos
#                     if char_2 != char
#                 )
#                 <= self.char_pos_assigns[char, pos],
#                 f"QuadCharacterAssignedLEQThanPosition({char},{pos},{pos_2})",
#             )
#             self.model.addCons(
#                 quicksum(
#                     self.quad_char_pos_assigns[char_2, char, pos_2, pos]
#                     for char_2 in self.chars.monos
#                     if char_2 != char
#                 )
#                 <= self.char_pos_assigns[char, pos],
#                 f"QuadCharacterAssignedLEQThanSecondPosition({char},{pos_2},{pos})",
#             )
#         for char_2 in self.chars.monos:
#             for pos in self.poss:
#                 if char_2 != char:
#                     self.model.addCons(
#                         quicksum(
#                             self.quad_char_pos_assigns[char, char_2, pos, pos_2]
#                             for pos_2 in self.poss
#                             if pos_2 != pos
#                         )
#                         <= self.char_pos_assigns[char, pos],
#                         f"QuadCharacterAssignedLEQThanCharacter({char},{char_2},"
#                         f"{pos})",
#                     )
#                     self.model.addCons(
#                         quicksum(
#                             self.quad_char_pos_assigns[char_2, char, pos_2, pos]
#                             for pos_2 in self.poss
#                             if pos_2 != pos
#                         )
#                         <= self.char_pos_assigns[char, pos],
#                         f"QuadCharacterAssignedLEQThanSecondCharacter({char_2},"
#                         f"{char},{pos})",
#                     )
#
#     for (char, char_2, pos, pos_2) in self.quad_char_pos_assigns_keys:
#         self.model.addCons(
#             self.char_pos_assigns[char, pos] + self.char_pos_assigns[char_2, pos_2]
#             <= 1 + self.quad_char_pos_assigns[char, char_2, pos, pos_2],
#             f"IntegrableQuadAssign({char},{pos_2},{pos})",
#         )
#
#     for pos in self.poss:
#         constr[pos] = self.model.addCons(
#             quicksum(self.char_pos_assigns[char, pos] for char in self.chars.monos)
#             == 1,
#             f"AllPositionsAssignedOnce({pos})",
#         )
#
#     self.model.setObjective(
#         quicksum(
#             costs * assigns
#             for (costs, assigns) in zip(
#                 self.char_pos_costs.values(), self.char_pos_assigns.values()
#             )
#         )
#         + quicksum(
#             costs * assigns
#             for (costs, assigns) in zip(
#                 self.quad_char_pos_costs.values(),
#                 self.quad_char_pos_assigns.values(),
#             )
#         ),
#         "minimize",
#     )
#
# def solve(self, visualize: bool = True) -> Optional[str]:
#     """Actually solve the optimization problem
#
#     Parameters
#     ----------
#     visualize : bool, optional
#         If True (default), the result will be shown after finishing
#     """
#     self.model.optimize()
#     if visualize:
#         return self.visualize_solution()
#
# def visualize_solution(self):
#     """Rudimentary visualize the optimization result on the console"""
#     solution_assignments = []
#     for (char, pos) in self.char_pos_assigns_keys:
#         print(
#             f"({char}, {pos}): "
#             f"{self.model.getVal(self.char_pos_assigns[char, pos])}, "
#             f"cost: {self.char_pos_costs[char, pos]}"
#         )
#         if self.model.getVal(self.char_pos_assigns[char, pos]) == 1:
#             solution_assignments.append((char, pos))
#     return str(solution_assignments)
#
# @property
# def char_pos_assigns_keys(self) -> Iterable[CharPosPair]:
#     """An iterator for the pairs of characters and corresponding positions"""
#     return product(self.chars.monos, self.poss)
#
# @property
# def quad_char_pos_assigns_keys(self) -> Iterable[CharPosQuadruple]:
#     """An iterator for quadruples of char. pairs and corresponding position pairs"""
#     flattened_tuple_of_quads = chain.from_iterable(
#         chain.from_iterable(
#             product(permutations(self.chars.monos, 2), permutations(self.poss, 2))
#         )
#     )
#     iter_of_quads = (iter(flattened_tuple_of_quads),) * 4
#     return zip(*iter_of_quads)
#
# @property
# def pos_pairs(self) -> Iterable:
#     """An iterator for all pairs of positions that are possible"""
#     return permutations(self.poss, 2)
