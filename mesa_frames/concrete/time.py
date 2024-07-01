"""
Mesa Time Module
================

Objects for handling the time component of a model. In particular, this module
contains Schedulers, which handle agent activation. A Scheduler is an object
which controls when agents are called upon to act, and when.

The activation order can have a serious impact on model behavior, so it's
important to specify it explicitly. Example simple activation regimes include
activating all agents in the same order every step, shuffling the activation
order every time, activating each agent *on average* once per step, and more.

Key concepts:
    Step: Many models advance in 'steps'. A step may involve the activation of
    all agents, or a random (or selected) subset of them. Each agent in turn
    may have their own step() method.

    Time: Some models may simulate a continuous 'clock' instead of discrete
    steps. However, by default, the Time is equal to the number of steps the
    model has taken.
"""

# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Self

import polars as pl

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.abstract.mixin import CopyMixin
from mesa_frames.concrete.agents import AgentsDF
from mesa_frames.concrete.model import ModelDF
from mesa_frames.types import IdsLike, Series, TimeT

# BaseScheduler has a self.time of int, while
# StagedActivation has a self.time of float


class BaseScheduler(CopyMixin):
    model: ModelDF
    steps: int
    time: TimeT
    _agents: AgentsDF
    _active_ids: pl.Series
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("copy", ["deep", "memo"]),
    }
    _copy_only_reference: list[str] = ["_model"]
    _original_step: Callable

    """
    A simple scheduler that activates AgentSet one at a time, in the order they were added.
    It assumes that each AgentSet added has a `step` method which takes no arguments and executes the AgentSet actions.

    Attributes:
        - model (ModelDF): The model instance associated with the scheduler.
        - steps (int): The number of steps the scheduler has taken.
        - time (TimeT): The current time in the simulation. Can be an integer or a float.

    Methods:
        - add: Adds an agent to the scheduler.
        - remove: Removes an agent from the scheduler.
        - step: Executes a step, which involves activating each agent once.
        - get_agent_count: Returns the number of agents in the scheduler.
        - agents (property): Returns a list of all agent instances.
    """

    def __init__(
        self, model: ModelDF, agentsets: Iterable[AgentSetDF] | None = None
    ) -> None:
        """Create a new BaseScheduler.

        Parameters:
        ----------
        model: ModelDF
            The model object associated with the scheduler.
        agents: Iterable[AgentSetDF], None, optional
            An iterable of agents to be scheduled. If None, defaults to an empty list.
        """
        self.model = model
        self.steps = 0
        self.time: TimeT = 0

        # Wrap the step to update model time
        self._original_step = self.step
        self.step = self._wrapped_step

        if agentsets is None:
            agentsets = []

        self._agents = AgentsDF()
        self._agents.add(agentsets)

    def add(
        self,
        agentsets: IdsLike | AgentSetDF | Iterable[AgentSetDF],
        inplace: bool = True,
    ) -> Self:
        """Add an agentsets or agents specified by their unique_id to the schedule.

        Parameters:
        ----------
        agentset: IdsLike | AgentSetDF | Iterable[AgentSetDF]
            An AgentSetDF object to be added to the scheduler. NOTE: The agentset must have a `step` method.

        Raises:
        -------
        ValueError
            If the agentset is already in the scheduler or the ids are already active.
        """
        obj = self._get_obj(inplace)
        if isinstance(agentsets, AgentSetDF) or (
            isinstance(agentsets, Iterable) and next(iter(agentsets), AgentSetDF)
        ):
            old_ids = obj._agents._ids.clone()
            obj._agents.add(agentsets, inplace=True)  # type: ignore (agentsets can be IdsLike according to PyLance)
            # Add only old active agents and new agents to active agents
            obj._active_ids = pl.concat(
                [
                    obj._active_ids,
                    obj._agents._ids.filter(obj._agents._ids.is_in(old_ids).not_()),
                ]
            )
        else:  # IdsLike
            ids = pl.Series(agentsets)
            if ids.is_in(self._active_ids).any():
                raise ValueError("Some ids are already active in the schedule")
            obj._active_ids = pl.concat([obj._active_ids, ids])
        return obj

    def remove(
        self, agents: IdsLike | AgentSetDF | Iterable[AgentSetDF], inplace: bool = True
    ) -> Self:
        """Remove ids or agentsets from the schedules.

        Note:
            It is only necessary to explicitly remove agents ids from the schedule if
            the agent is not removed from the agentset.

        Parameters:
        ----------
        agents: IdsLike | AgentSetDF | Iterable[AgentSetDF]
            The ids or agentsets to be removed from the schedule.
        inplace: bool, optional
            If True, the operation is performed in place. Otherwise, a new object is returned.
        """
        obj = self._get_obj(inplace)
        if isinstance(agents, AgentSetDF):
            obj._agents._agentsets.remove(agents)
        else:
            obj._active_ids = obj._active_ids.filter(self._active_ids != agents)
        return obj

    def step(self, inplace: bool = True) -> Self:
        """Execute the step of all the AgentSetDF in the schedule, one at a time, in the order they were added.

        Parameters:
        ----------
        inplace: bool, optional
            If True, the operation is performed in place. Otherwise, a new object is returned.

        Returns:
        -------
        Self
        """
        obj = self._get_obj(inplace)
        obj.do_each("step", shuffle=False, inplace=True)
        obj.steps += 1
        obj.time += 1
        return obj

    def _wrapped_step(self, inplace: bool = True) -> Self:
        """Wrapper for the step method to include time and step updating."""
        obj = self._get_obj(inplace)
        obj._original_step(inplace=True)
        obj.model._advance_time()
        return obj

    def get_agent_count(self) -> int:
        """Returns the current number of agents in the queue."""
        return len(self._active_ids)

    def get_agent_keys(self, shuffle: bool = False) -> Series:
        """Return the ids of all agents in the scheduler.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the order of , by default False

        Returns
        -------
        Series
            _description_
        """
        agent_keys = self._agents._ids
        if shuffle:
            self.model.random.shuffle(agent_keys)
        return agent_keys

    def do_each(self, method: str, shuffle: bool = False, inplace: bool = True) -> Self:
        """Execute the method on each AgentSet.

        Parameters
        ----------
        method : str
            _description_
        shuffle : bool, optional
            Whether to shuffle the order of AgentSet, by default False
        inplace : bool, optional
            Whether to modify the AgentSet in place, by default True

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace)
        if shuffle:
            self._agents.shuffle(inplace=True)
        self._agents.do(method, inplace=True)
        return obj

    @property
    def agents(self) -> AgentsDF:
        return self._agents

    @property
    def active_agents(self) -> AgentsDF:
        return self._agents.select(self._active_ids, inplace=False)

    def __len__(self) -> int:
        return len(self._active_ids)


class RandomActivation(BaseScheduler):
    """
    A scheduler that activates each AgentSetDF once per step, in a random order, with the order reshuffled each step.

    This scheduler is equivalent to the NetLogo 'ask agents...' behavior and is a common default for ABMs.
    It assumes that all AgentSetDF have a `step` method.

    The random activation ensures that no single agent or sequence of agents consistently influences the model due
    to ordering effects, which is crucial for certain types of simulations.

    Inherits all attributes and methods from BaseScheduler.

    Methods:
    --------
    step(inplace: bool = True) -> Self
        Executes the step of all AgentSetDF, one at a time, in random order.
    """

    def step(self, inplace: bool = True) -> Self:
        """Executes the step of all agents, one at a time, in
        random order.

        """
        obj = self._get_obj(inplace)
        obj.do_each("step", shuffle=True, inplace=True)
        obj.steps += 1
        obj.time += 1
        return obj


class SimultaneousActivation(BaseScheduler):
    """
    A scheduler that simulates the simultaneous activation of all AgentSetDF.

    This scheduler is unique in that it requires AgentSetDFs to have both `step` and `advance` methods.
    - The `step` method is for activating the agent and staging any changes without applying them immediately.
    - The `advance` method then applies these changes, simulating simultaneous action.

    This scheduler is useful in scenarios where the interactions between agents are sensitive to the order
    of execution, and a quasi-simultaneous execution is more realistic.

    Inherits all attributes and methods from BaseScheduler.

    Methods:
    step(inplace: bool = True) -> Self
        Executes a step for all agents, first calling `step` then `advance` on each.
    """

    def step(self, inplace: bool = True) -> Self:
        """Step all agents, then advance them."""
        obj = self._get_obj(inplace)
        obj.do_each("step", shuffle=False, inplace=True)
        obj.do_each("advance", shuffle=False, inplace=True)
        obj.steps += 1
        obj.time += 1
        return obj


class StagedActivation(BaseScheduler):
    """
    A scheduler allowing agent activation to be divided into several stages, with all AgentSetDF executing one stage
    before moving on to the next. This class is a generalization of SimultaneousActivation.

    This scheduler is useful for complex models where actions need to be broken down into distinct phases
    for each AgentSetDF in each time step. AgentSetDFs must implement methods for each defined stage.

    The scheduler also tracks steps and time separately, allowing fractional time increments based on the number
    of stages. Time advances in fractional increments of 1 / (# of stages), meaning that 1 step = 1 unit of time.

    Inherits all attributes and methods from BaseScheduler.

    Attributes:
    ----------
    stage_list : list[str]
        A list of stage names that define the order of execution.
    shuffle : bool
        Determines whether to shuffle the order of agents each step.
    shuffle_between_stages : bool
        Determines whether to shuffle agents between each stage.

    Methods:
    -------
    step(inplace: bool = True) -> Self
        Executes all the stages for all AgentSetDFs in the defined order.
    """

    def __init__(
        self,
        model: ModelDF,
        agentsets: Iterable[AgentSetDF] | None = None,
        stage_list: list[str] | None = None,
        shuffle: bool = False,
        shuffle_between_stages: bool = False,
    ) -> None:
        """Create an empty Staged Activation schedule.

        Parameters:
        ----------
        model: ModelDF
            The model to which the schedule belongs.
        agentsets: Iterable[AgentSetDF], None, optional
            An iterable of AgentSetDF who are controlled by the schedule.
        stage_list: list[str], None, optional
            A list of strings of names of stages to run, in the order to run them in.
        shuffle: bool, optional
            If True, shuffle the order of agents each step.
        shuffle_between_stages: bool, optional
            If True, shuffle the agents after each stage; otherwise, only shuffle at the start of each step.
        """
        super().__init__(model, agentsets)
        self.stage_list = stage_list if stage_list else ["step"]
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages
        self.stage_time = 1 / len(self.stage_list)

    def step(self, inplace: bool = True) -> Self:
        """Executes all the stages for all agents.

        Parameters:
        ----------
        inplace: bool, optional
            If True, the operation is performed in place. Otherwise, a new object is returned.

        Returns:
        -------
        Self
        """
        obj = self._get_obj(inplace)
        shuffle = obj.shuffle
        for stage in obj.stage_list:
            if stage.startswith("model."):
                getattr(obj.model, stage[6:])()
            else:
                obj.do_each(stage, shuffle=shuffle, inplace=True)

            shuffle = obj.shuffle_between_stages
            obj.time += obj.stage_time

        obj.steps += 1
        return obj


class RandomActivationByType(BaseScheduler):
    """
    A scheduler that activates each type of agent once per step, in random order, with the order reshuffled every step.

    This scheduler is useful for models with multiple types of agents, ensuring that each type is treated
    equitably in terms of activation order. The randomness in activation order helps in reducing biases
    due to ordering effects.

    Inherits all attributes and methods from BaseScheduler.

    If you want to do some computations / data collections specific to an AgentSetDF
    type, you can either:
    - access via scheduler.agentsets_by_type or model.agents.agentsets_by_type
    - loop through all AgentSetDF, and filter by their type

    Methods:
    -------
    step()
        Executes the step of each agent type in a random order.
        - step_type: Activates all agents of a given type.
        - get_type_count: Returns the count of agents of a specific type.

    Properties:
    ----------
    agentsets_by_type : dict[type[AgentSetDF], list[AgentSetDF]]
        A dictionary mapping AgentSetDF types to AgentSetDFs.
    """

    def step(
        self,
        shuffle_types: bool = True,
        shuffle_agents: bool = True,
        inplace: bool = True,
    ) -> Self:
        """
        Executes the step of each agent type, one at a time, in random order.

        Parameters:
        ----------
        shuffle_types : bool, optional
            If True, the order of execution of each types is shuffled.
        shuffle_agents : bool, optional
            If True, the order of execution of each agents in a type group is shuffled.
        inplace : bool, optional
            If True, the operation is performed in place. Otherwise, a new object is returned.
        """
        obj = self._get_obj(inplace)
        type_keys = obj.agentsets_by_type.keys()
        if shuffle_types:
            obj.model.random.shuffle(type_keys)
        for agent_class in type_keys:
            obj.step_type(agent_class, shuffle_agents=shuffle_agents, inplace=True)
        obj.steps += 1
        obj.time += 1
        return obj

    def step_type(
        self,
        agentset_type: type[AgentSetDF],
        shuffle_agents: bool = True,
        inplace: bool = True,
    ) -> Self:
        """
        Shuffle order and run all agents of a given type.
        This method is equivalent to the NetLogo 'ask [breed]...'.

        Parameters:
        ----------
        agentset_type : type[AgentSetDF]
            Class object of the type to run.
        """
        agents = self.agentsets_by_type[agentset_type]
        if shuffle_agents:
            agents.shuffle(inplace=True)
        agents.do("step")

    def get_type_count(self, agentset_type: type[AgentSetDF]) -> int:
        """
        Returns the current number of agents of certain type in the queue.
        """
        return len(self.agentsets_by_type[agentset_type])

    @property
    def agentsets_by_type(self) -> dict[type[AgentSetDF], list[AgentSetDF]]:
        return self._agents.agentsets_by_type


class DiscreteEventScheduler(BaseScheduler):
    """
    This class has been deprecated and replaced by the functionality provided by experimental.devs
    """

    def __init__(self, model: ModelDF, time_step: TimeT = 1) -> None:
        """

        Args:
            model (ModelDF): The model to which the schedule belongs
            time_step (TimeT): The fixed time step between steps

        """
        super().__init__(model)
        raise Exception(
            "DiscreteEventScheduler is deprecated in favor of the functionality provided by experimental.devs"
        )
