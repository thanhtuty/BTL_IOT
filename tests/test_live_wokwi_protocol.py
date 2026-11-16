"""Protocol checks using synthetic numbers only (no private photos or models)."""
import unittest

from live.live_wokwi import protocol_message


class LiveWokwiProtocolTest(unittest.TestCase):
    def test_round_trip_mqtt_payload(self):
        self.assertEqual(protocol_message(3, 0.564071, 7), "P,3,5641,7")
        self.assertEqual(protocol_message(4, -0.065, 8), "P,4,-650,8")

    def test_min_max_cosine(self):
        self.assertEqual(protocol_message(1, -1.0, 1), "P,1,-10000,1")
        self.assertEqual(protocol_message(4, 1.0, 2), "P,4,10000,2")

    def test_reject_outside_supported_classes(self):
        with self.assertRaises(ValueError):
            protocol_message(0, 0.5, 2)
        with self.assertRaises(ValueError):
            protocol_message(5, 0.5, 2)

    def test_reject_wrong_similarity_and_sequence(self):
        for cosine in (1.01, -1.01, float("nan")):
            with self.assertRaises(ValueError):
                protocol_message(2, cosine, 3)
        with self.assertRaises(ValueError):
            protocol_message(2, 0.4, 0)


if __name__ == "__main__":
    unittest.main()
