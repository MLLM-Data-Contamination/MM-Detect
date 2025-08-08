"""Resume management utilities for MM-Detect."""

import json
import os
import hashlib
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import get_config

@dataclass
class ResumeState:
    """Resume state information."""
    task_id: str
    method: str
    model_name: str
    dataset_name: str
    total_items: int
    completed_items: int
    failed_items: List[int]
    results: Dict[str, Any]
    checkpoint_file: str
    timestamp: str
    progress_percentage: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResumeState':
        """Create ResumeState from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ResumeState to dictionary."""
        return asdict(self)

class ResumeManager:
    """Manager for handling resume functionality in MM-Detect."""
    
    def __init__(self, method: str, model_name: str, dataset_name: str, 
                 output_dir: Optional[str] = None):
        """Initialize Resume Manager.
        
        Args:
            method: Detection method name
            model_name: Model name being tested
            dataset_name: Dataset name
            output_dir: Output directory (optional, from config if not provided)
        """
        self.config = get_config()
        self.method = method
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir or self.config.output_dir)
        
        # Generate unique task ID
        self.task_id = self._generate_task_id()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define checkpoint file paths
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / f"{self.task_id}.json"
        self.results_file = self.output_dir / "results.json"
        
        self.resume_state: Optional[ResumeState] = None
        
    def _generate_task_id(self) -> str:
        """Generate unique task ID based on method, model, and dataset."""
        content = f"{self.method}_{self.model_name}_{self.dataset_name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def can_resume(self) -> bool:
        """Check if resume is possible.
        
        Returns:
            True if checkpoint file exists and is valid.
        """
        if not self.checkpoint_file.exists():
            return False
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Basic validation
                required_fields = ['task_id', 'method', 'model_name', 'dataset_name', 'results']
                return all(field in data for field in required_fields)
        except (json.JSONDecodeError, KeyError, OSError):
            return False
    
    def load_checkpoint(self) -> Optional[ResumeState]:
        """Load checkpoint from file.
        
        Returns:
            ResumeState if successful, None otherwise.
        """
        if not self.can_resume():
            return None
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.resume_state = ResumeState.from_dict(data)
                return self.resume_state
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None
    
    def save_checkpoint(self, total_items: int, completed_items: int, 
                       results: Dict[str, Any], failed_items: List[int] = None):
        """Save current progress to checkpoint.
        
        Args:
            total_items: Total number of items to process
            completed_items: Number of completed items
            results: Current results dictionary
            failed_items: List of indices that failed processing
        """
        failed_items = failed_items or []
        progress_percentage = (completed_items / total_items * 100) if total_items > 0 else 0
        
        resume_state = ResumeState(
            task_id=self.task_id,
            method=self.method,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            total_items=total_items,
            completed_items=completed_items,
            failed_items=failed_items,
            results=results,
            checkpoint_file=str(self.checkpoint_file),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            progress_percentage=progress_percentage
        )
        
        try:
            # Save checkpoint
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(resume_state.to_dict(), f, ensure_ascii=False, indent=2)
            
            # Also save results to main results file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            self.resume_state = resume_state
            
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def get_completed_indices(self) -> set:
        """Get set of completed item indices.
        
        Returns:
            Set of completed indices.
        """
        if self.resume_state is None:
            return set()
            
        # Extract completed indices from results
        completed = set()
        if 'processed_indices' in self.resume_state.results:
            completed.update(self.resume_state.results['processed_indices'])
        
        # Also check individual result entries
        for key in self.resume_state.results.keys():
            if key.startswith('item_') and key[5:].isdigit():
                completed.add(int(key[5:]))
                
        return completed
    
    def get_failed_indices(self) -> set:
        """Get set of failed item indices.
        
        Returns:
            Set of failed indices.
        """
        if self.resume_state is None:
            return set()
        return set(self.resume_state.failed_items)
    
    def mark_item_processed(self, index: int, result: Any = None):
        """Mark an item as processed.
        
        Args:
            index: Item index
            result: Processing result (optional)
        """
        if self.resume_state is None:
            return
            
        # Add to processed indices
        if 'processed_indices' not in self.resume_state.results:
            self.resume_state.results['processed_indices'] = []
        
        if index not in self.resume_state.results['processed_indices']:
            self.resume_state.results['processed_indices'].append(index)
        
        # Store result if provided
        if result is not None:
            self.resume_state.results[f'item_{index}'] = result
            
        # Update completed count
        self.resume_state.completed_items = len(self.resume_state.results['processed_indices'])
    
    def mark_item_failed(self, index: int, error: str = None):
        """Mark an item as failed.
        
        Args:
            index: Item index
            error: Error message (optional)
        """
        if self.resume_state is None:
            return
            
        if index not in self.resume_state.failed_items:
            self.resume_state.failed_items.append(index)
            
        # Store error information
        if error:
            if 'errors' not in self.resume_state.results:
                self.resume_state.results['errors'] = {}
            self.resume_state.results['errors'][str(index)] = error
    
    def print_resume_info(self):
        """Print resume information to console."""
        if self.resume_state is None:
            print("No checkpoint found. Starting from beginning.")
            return
            
        print(f"\n🔄 Resuming from checkpoint:")
        print(f"   Task ID: {self.resume_state.task_id}")
        print(f"   Method: {self.resume_state.method}")
        print(f"   Model: {self.resume_state.model_name}")
        print(f"   Dataset: {self.resume_state.dataset_name}")
        print(f"   Progress: {self.resume_state.completed_items}/{self.resume_state.total_items} "
              f"({self.resume_state.progress_percentage:.1f}%)")
        print(f"   Failed items: {len(self.resume_state.failed_items)}")
        print(f"   Last saved: {self.resume_state.timestamp}")
        print(f"   Checkpoint: {self.checkpoint_file}\n")
    
    def cleanup_checkpoint(self):
        """Clean up checkpoint file after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                print(f"✅ Cleaned up checkpoint file: {self.checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to cleanup checkpoint: {e}")
    
    def get_resume_statistics(self) -> Dict[str, Any]:
        """Get resume statistics.
        
        Returns:
            Dictionary with resume statistics.
        """
        if self.resume_state is None:
            return {}
            
        return {
            'can_resume': True,
            'task_id': self.resume_state.task_id,
            'progress_percentage': self.resume_state.progress_percentage,
            'completed_items': self.resume_state.completed_items,
            'total_items': self.resume_state.total_items,
            'failed_items': len(self.resume_state.failed_items),
            'timestamp': self.resume_state.timestamp
        }

def create_resume_manager(method: str, model_name: str, dataset_name: str, 
                         output_dir: Optional[str] = None) -> ResumeManager:
    """Create and return a ResumeManager instance.
    
    Args:
        method: Detection method name
        model_name: Model name
        dataset_name: Dataset name
        output_dir: Output directory (optional)
        
    Returns:
        ResumeManager instance.
    """
    return ResumeManager(method, model_name, dataset_name, output_dir)