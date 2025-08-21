import requests

def get_black_move(fen):
        """Query online tablebase (Lichess API)"""
        url = f"http://tablebase.lichess.ovh/standard?fen={fen}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'moves' in data and data['moves']:
                # Get the best move (first in the list)
                best_move_data = data['moves'][0]
                return best_move_data['uci']  # Return UCI string directly
            return None
        except:
            return None