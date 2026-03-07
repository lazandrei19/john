from romanian_whist.rules.cards import card_label
from romanian_whist.rules.config import OneCardMode, WhistVariantConfig
from romanian_whist.rules.game import RomanianWhistGame, action_from_bid, action_from_card


def test_schedule_for_four_players() -> None:
    config = WhistVariantConfig(players=4)
    assert config.schedule() == (
        8,
        8,
        8,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        8,
        8,
        8,
        8,
    )


def test_one_card_modes_cycle() -> None:
    config = WhistVariantConfig(players=4, one_card_modes=(OneCardMode.REGULAR, OneCardMode.FOREHEAD, OneCardMode.BLIND))
    assert config.one_card_mode_for_index(0) == OneCardMode.REGULAR
    assert config.one_card_mode_for_index(1) == OneCardMode.FOREHEAD
    assert config.one_card_mode_for_index(2) == OneCardMode.BLIND
    assert config.one_card_mode_for_index(3) == OneCardMode.REGULAR


def test_card_label_works() -> None:
    assert card_label(0) == "2C"
    assert card_label(51) == "AS"


def test_last_bid_restriction_is_enforced() -> None:
    game = RomanianWhistGame(WhistVariantConfig(players=4, seed=7))
    game.reset(seed=7)
    order = game.round_state.bidding_order
    for seat, bid in zip(order[:-1], [2, 2, 1]):
        assert seat == game.current_player
        game.step(action_from_bid(bid))
    last_player = game.current_player
    legal_bids = [action for action in game.legal_actions(last_player)]
    assert action_from_bid(3) not in legal_bids


def test_hidden_one_card_observation() -> None:
    game = RomanianWhistGame(WhistVariantConfig(players=4, seed=3, one_card_modes=(OneCardMode.FOREHEAD,)))
    game.reset(seed=3)
    for _ in range(10):
        if game.round_state.hand_size == 1:
            break
        while game.round_state.phase == "bidding":
            game.step(game.legal_actions()[0])
        while not game.match_finished and game.round_state.phase == "play":
            game.step(game.legal_actions()[0])
    observation = game.observe(0)
    assert observation["hand_mask"].sum() == 0
    assert any(card >= 0 for card in observation["public_card_by_player"][1:4])


def test_trick_winner_and_round_scoring() -> None:
    game = RomanianWhistGame(WhistVariantConfig(players=3, seed=1))
    game.reset(seed=1)
    state = game.round_state
    state.hand_size = 1
    state.phase = "bidding"
    state.dealer = 2
    state.leader = 0
    state.turn_index = 0
    state.trump_suit = 3
    state.hands = [[51], [12], [0]]
    for bid in [1, 0, 1]:
        game.step(action_from_bid(bid))
    game.step(action_from_card(51))
    game.step(action_from_card(12))
    outcome = game.step(action_from_card(0))
    assert outcome.round_finished
    assert game.scores[0] == 6
