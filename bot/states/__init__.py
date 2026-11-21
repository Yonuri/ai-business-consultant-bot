"""FSM-состояния."""

from aiogram.fsm.state import State, StatesGroup


class AdminStates(StatesGroup):
    waiting_document = State()
    waiting_url = State()
    waiting_test_question = State()


class UserStates(StatesGroup):
    waiting_manager_request = State()
