"""
Offline functionality handler for MalariaAI
Manages data storage and synchronization when offline
"""

import json
import os
from datetime import datetime
import streamlit as st

class OfflineHandler:
    def __init__(self, offline_file='offline_predictions.json'):
        self.offline_file = offline_file
        
    def save_prediction(self, prediction_data):
        """Save prediction data for offline use"""
        try:
            # Load existing offline data
            offline_data = self.load_offline_data()
            
            # Add timestamp and sync status
            prediction_data.update({
                'timestamp': datetime.now().isoformat(),
                'synced': False,
                'id': len(offline_data) + 1
            })
            
            # Append new prediction
            offline_data.append(prediction_data)
            
            # Save back to file
            with open(self.offline_file, 'w') as f:
                json.dump(offline_data, f, indent=2)
                
            return True
        except Exception as e:
            st.error(f"Failed to save offline data: {str(e)}")
            return False
    
    def load_offline_data(self):
        """Load offline prediction data"""
        try:
            if os.path.exists(self.offline_file):
                with open(self.offline_file, 'r') as f:
                    return json.load(f)
            return []
        except:
            return []
    
    def sync_data(self):
        """Mark all offline data as synced"""
        try:
            offline_data = self.load_offline_data()
            
            # Mark all as synced
            for item in offline_data:
                item['synced'] = True
                item['sync_timestamp'] = datetime.now().isoformat()
            
            # Save updated data
            with open(self.offline_file, 'w') as f:
                json.dump(offline_data, f, indent=2)
                
            return len(offline_data)
        except:
            return 0
    
    def get_unsynced_count(self):
        """Get count of unsynced predictions"""
        offline_data = self.load_offline_data()
        return len([item for item in offline_data if not item.get('synced', False)])
    
    def clear_synced_data(self):
        """Remove synced data to save space"""
        try:
            offline_data = self.load_offline_data()
            unsynced_data = [item for item in offline_data if not item.get('synced', False)]
            
            with open(self.offline_file, 'w') as f:
                json.dump(unsynced_data, f, indent=2)
                
            return len(offline_data) - len(unsynced_data)
        except:
            return 0

def check_connection():
    """Simple connection check"""
    try:
        import urllib.request
        urllib.request.urlopen('http://www.google.com', timeout=1)
        return True
    except:
        return False

def display_offline_status():
    """Display offline status and data in sidebar"""
    offline_handler = OfflineHandler()
    
    # Check connection
    is_online = check_connection()
    
    if is_online:
        st.sidebar.success("🟢 Online")
        
        # Show sync option if there's unsynced data
        unsynced_count = offline_handler.get_unsynced_count()
        if unsynced_count > 0:
            if st.sidebar.button(f"🔄 Sync {unsynced_count} predictions"):
                synced = offline_handler.sync_data()
                st.sidebar.success(f"✅ Synced {synced} predictions")
                st.rerun()
    else:
        st.sidebar.warning("🔴 Offline Mode")
        st.sidebar.info("📱 Data will be saved locally")
    
    # Show offline data summary
    offline_data = offline_handler.load_offline_data()
    if offline_data:
        with st.sidebar.expander(f"📊 Offline Data ({len(offline_data)})"):
            for item in offline_data[-3:]:  # Show last 3
                status = "✅" if item.get('synced') else "⏳"
                timestamp = item.get('timestamp', 'Unknown')[:16]
                st.write(f"{status} {timestamp}")
    
    return is_online, offline_handler