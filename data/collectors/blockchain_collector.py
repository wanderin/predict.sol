import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed

logger = logging.getLogger(__name__)

class BlockchainCollector:
    """Collects on-chain data from Solana blockchain."""
    
    def __init__(self, rpc_endpoint: str, metrics: List[str], update_interval: int):
        """
        Initialize the blockchain collector.
        
        Args:
            rpc_endpoint: Solana RPC endpoint URL
            metrics: List of metrics to collect
            update_interval: Data collection interval in seconds
        """
        self.client = AsyncClient(rpc_endpoint, commitment=Confirmed)
        self.metrics = metrics
        self.update_interval = update_interval
        self.latest_data: Dict = {}
        
    async def get_transaction_metrics(self, slot_range: int = 1000) -> Dict:
        """Collect transaction-related metrics."""
        try:
            # Get recent performance samples
            response = await self.client.get_recent_performance_samples(slot_range)
            
            if response["result"]:
                samples = response["result"]
                
                # Calculate average TPS and other metrics
                avg_tps = sum(sample["numTransactions"] / sample["samplePeriodSecs"] 
                            for sample in samples) / len(samples)
                
                return {
                    "avg_tps": avg_tps,
                    "num_transactions": sum(sample["numTransactions"] for sample in samples),
                    "num_slots": slot_range
                }
                
        except Exception as e:
            logger.error(f"Error collecting transaction metrics: {e}")
            return {}
            
    async def get_stake_metrics(self) -> Dict:
        """Collect staking-related metrics."""
        try:
            # Get stake activation data
            response = await self.client.get_stake_activation(commitment=Confirmed)
            
            if response["result"]:
                data = response["result"]
                return {
                    "active_stake": data["active"],
                    "inactive_stake": data["inactive"],
                    "activating_stake": data["activating"],
                    "deactivating_stake": data["deactivating"]
                }
                
        except Exception as e:
            logger.error(f"Error collecting stake metrics: {e}")
            return {}
            
    async def get_defi_metrics(self) -> Dict:
        """Collect DeFi-related metrics using external APIs."""
        try:
            async with aiohttp.ClientSession() as session:
                # Example: collecting TVL data from DefiLlama
                async with session.get("https://api.llama.fi/protocol/solana") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "tvl": data.get("tvl", 0),
                            "volume_24h": data.get("volume24h", 0),
                        }
        except Exception as e:
            logger.error(f"Error collecting DeFi metrics: {e}")
            return {}
    
    async def collect_data(self) -> Dict:
        """Collect all specified blockchain metrics."""
        tasks = []
        
        if "transaction_count" in self.metrics:
            tasks.append(self.get_transaction_metrics())
        if "active_stakes" in self.metrics:
            tasks.append(self.get_stake_metrics())
        if "defi_tvl" in self.metrics:
            tasks.append(self.get_defi_metrics())
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all metrics
        collected_data = {}
        for result in results:
            if isinstance(result, Dict):
                collected_data.update(result)
                
        self.latest_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": collected_data
        }
        
        return self.latest_data
    
    async def start_collection(self):
        """Start continuous data collection."""
        while True:
            try:
                await self.collect_data()
                logger.info("Blockchain data collected successfully")
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in blockchain data collection: {e}")
                await asyncio.sleep(self.update_interval)
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest collected data."""
        return self.latest_data if self.latest_data else None