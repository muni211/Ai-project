# File: core/ai_engine.py
# ליבת ה-AI הבסיסית – נבנה רשת נוירונים פשוטה בעזרת NumPy

import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # אתחול משקלים לשכבת הקלט לשכבת ההסתר
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # אתחול משקלים לשכבת ההסתר לשכבת הפלט
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        # פונקציית ההפעלה ReLU – מחזירה את הערך אם חיובי, ואפס אם לא.
        return np.maximum(0, x)
    
    def softmax(self, x):
        # softmax נותן לנו הסתברויות – כך שמיד הפלט יהיה בין 0 ל-1.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # מעבר קדמי לרשת – חישובים בין השכבות
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def train_on_batch(self, X, y, epochs=10):
        # אימון המודל על קבוצת נתונים (batch) – נעשה זאת על ידי מעבר קדמי וחישוב שגיאה ועדכון משקלים
        m = X.shape[0]
        for epoch in range(epochs):
            # מעבר קדמי
            output = self.forward(X)
            
            # הפיכת התוויות לפורמט one-hot
            y_onehot = np.zeros_like(output)
            y_onehot[np.arange(m), y] = 1
            
            # חישוב שגיאה (Loss) – נשתמש ב-cross-entropy
            loss = -np.sum(y_onehot * np.log(output + 1e-8)) / m
            
            # חישוב השיפועים (gradient) והעדכון של המשקלים
            dZ2 = (output - y_onehot) / m
            dW2 = np.dot(self.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * (self.Z1 > 0)  # נגזרת ReLU
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # עדכון המשקלים
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

if __name__ == '__main__':
    # דוגמה להרצת המודל:
    input_size = 10   # מספר ערכים שמוזנים
    hidden_size = 16  # מספר נוירונים בשכבת ההסתר
    output_size = 3   # מספר קטגוריות (למשל, תוויות 0, 1, 2)
    num_samples = 100
    
    np.random.seed(42)  # לשמירה על תוצאות עקביות
    X = np.random.rand(num_samples, input_size)
    y = np.random.randint(0, output_size, size=num_samples)
    
    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    nn.train_on_batch(X, y, epochs=20)
    predictions = nn.forward(X[:5])
    print("Predictions on first 5 samples:", predictions)
