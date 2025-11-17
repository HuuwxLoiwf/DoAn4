import numpy as np
from PIL import Image as PILImage
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from .models import Image as ImageModel

class OptimizedAISearchService:
    """Optimized AI Image Search Service with caching"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load model once - singleton pattern"""
        if self._model is None:
            self._model = ResNet50(
                weights='imagenet', 
                include_top=False, 
                pooling='avg'
            )
    
    @property
    def model(self):
        return self._model
    
    def get_feature_vector(self, img):
        """Extract feature vector from image"""
        if isinstance(img, str):
            img = PILImage.open(img).convert('RGB')
        elif not isinstance(img, PILImage.Image):
            img = PILImage.open(img).convert('RGB')
        
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = self.model.predict(img_array, verbose=0)
        return features.flatten()
    
    def calculate_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def search_similar_images(self, query_image, threshold=0.7, limit=10, exclude_id=None):
        """
        Search similar images with cached vectors
        """
        # Extract query vector
        query_vector = self.get_feature_vector(query_image)
        
        # Get images with cached vectors
        images_with_vectors = ImageModel.objects.exclude(
            feature_vector__isnull=True
        )
        
        if exclude_id:
            images_with_vectors = images_with_vectors.exclude(id=exclude_id)
        
        results = []
        
        for img_obj in images_with_vectors:
            try:
                # Use cached vector
                db_vector = np.array(img_obj.feature_vector)
                
                # Calculate similarity
                similarity = self.calculate_similarity(query_vector, db_vector)
                
                if similarity >= threshold:
                    results.append((img_obj, float(similarity)))
                    
            except Exception as e:
                print(f"Error processing image {img_obj.id}: {str(e)}")
                continue
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def process_and_save_vector(self, image_obj):
        """Process image and save vector to database"""
        try:
            # Load image from local file
            img = PILImage.open(image_obj.image_file.path).convert('RGB')
            
            # Extract feature vector
            vector = self.get_feature_vector(img)
            
            # Save to database
            image_obj.feature_vector = vector.tolist()
            image_obj.save(update_fields=['feature_vector'])
            
            return True
            
        except Exception as e:
            print(f"Error processing vector for image {img_obj.id}: {str(e)}")
            return False

# Singleton instance
ai_search = OptimizedAISearchService()