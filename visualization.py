import matplotlib.pyplot as plt
import numpy as np

def create_epic_visualization(history):
    """Create a visualization of the training history."""
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()

def plot_predictions(y_true, y_pred, r2, rmse, mae):
    """Create a scatter plot of predictions vs actual values."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    
    plt.title(f'Predictions vs Actual Values\nR² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.grid(True)
    
    return plt.gcf() 