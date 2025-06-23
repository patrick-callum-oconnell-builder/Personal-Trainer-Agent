# AI Personal Trainer

An AI-powered personal trainer application that helps users with their fitness goals through intelligent automation and Google service integration. The application can schedule workouts, manage meal planning, provide fitness advice, and automate various fitness-related tasks to save users time.

## 🎥 Demo

Here is a video demo of the functionality (in case you don't want to set up all of the Google Service API enabling, which isn't yet automated): 
https://drive.google.com/file/d/10pUME3WR3DRclbFq2YwOQmOTCEIaif9s/view?usp=sharing 

## ✨ Features

- **Real-time AI Conversation**: Chat with an intelligent personal trainer that understands context and provides personalized advice
- **Google Services Integration**: 
  - 📅 **Google Calendar**: Schedule and manage workouts automatically
  - 📧 **Gmail**: Send workout reminders and meal plans
  - 🗺️ **Google Maps**: Find nearby gyms, parks, and workout locations
  - 📁 **Google Drive**: Store and manage fitness documents and meal plans
  - 📊 **Google Sheets**: Track progress and maintain fitness logs
  - ✅ **Google Tasks**: Create and manage fitness-related tasks
  - 📱 **Google Fit**: Sync fitness data and track activities
- **State-Based Agent Orchestration**: Intelligent decision-making for when to chat, use tools, or record preferences
- **Knowledge Graph**: Maintains context and user preferences across conversations
- **Modern Web Interface**: Clean React/MUI frontend with real-time chat and knowledge graph visualization

## 🏗️ Architecture

### Tech Stack
- **Backend**: Python FastAPI with async/await support
- **Agent Framework**: LangGraph for state-based orchestration
- **AI/LLM**: OpenAI GPT models via LangChain
- **Frontend**: React 18 with Material-UI (MUI)
- **State Management**: Custom thread-safe state management with validation
- **Testing**: pytest with comprehensive unit and integration tests

### Project Structure
```
agent_personal_trainer/
├── backend/                    # FastAPI backend application
│   ├── agent.py               # Main agent orchestration logic
│   ├── agent_state_machine.py # State machine for agent workflow
│   ├── agent_state.py         # Thread-safe state management
│   ├── dictionary_state.py    # Base state management class
│   ├── knowledge_graph.py     # Knowledge graph implementation
│   ├── tool_manager.py        # Tool execution and management
│   ├── prompts.py             # AI prompts and system messages
│   ├── time_formatting.py     # Time parsing utilities
│   ├── auth.py                # Google authentication
│   ├── main.py                # FastAPI application entry point
│   ├── api/                   # API routes and endpoints
│   ├── google_services/       # Google API integrations
│   │   ├── calendar.py        # Google Calendar integration
│   │   ├── gmail.py           # Gmail integration
│   │   ├── maps.py            # Google Maps integration
│   │   ├── drive.py           # Google Drive integration
│   │   ├── sheets.py          # Google Sheets integration
│   │   ├── tasks.py           # Google Tasks integration
│   │   ├── fit.py             # Google Fit integration
│   │   └── auth.py            # Google authentication
│   ├── tools/                 # Agent tools
│   │   ├── calendar_tools.py  # Calendar-related tools
│   │   ├── maps_tools.py      # Maps and location tools
│   │   └── preferences_tools.py # User preference tools
│   └── tests/                 # Backend tests
│       ├── unit/              # Unit tests
│       └── integration/       # Integration tests
├── frontend/                  # React frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Chat.tsx       # Chat interface
│   │   │   └── KnowledgeGraph.tsx # Knowledge graph visualization
│   │   └── App.tsx            # Main application component
│   └── package.json           # Frontend dependencies
├── tests/                     # End-to-end tests
├── run.py                     # Main application launcher
├── setup_auth.py              # Google authentication setup
└── README.md                  # This file
```

### Agent Architecture

The AI agent uses a state machine with the following states:
- **Active**: Processing user input and making decisions
- **Awaiting User**: Waiting for user input
- **Awaiting Tool**: Executing a tool and waiting for results
- **Error**: Handling errors gracefully
- **Done**: Task completed

The agent can:
1. **Chat**: Provide fitness advice and answer questions
2. **Use Tools**: Execute Google service integrations
3. **Record Preferences**: Store user preferences in the knowledge graph
4. **Manage State**: Maintain conversation context and user data

## 📋 Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Node.js 14+** (18+ recommended)
- **Google Cloud Platform account** with the following APIs enabled:
  - Google Calendar API
  - Gmail API
  - Google Maps API
  - Google Drive API
  - Google Sheets API
  - Google Tasks API
  - Google Fit API
- **OpenAI API key** for AI/LLM functionality

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd agent_personal_trainer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_PROJECT_ID=your-google-project-id

# Optional (for enhanced features)
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
```

### 3. Setup Google Authentication

```bash
# Run the authentication setup script
python setup_auth.py
```

This will:
- Guide you through Google OAuth setup
- Authenticate all required Google services
- Store credentials securely

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Start the Application

```bash
# Start both backend and frontend
python run.py
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Detailed Setup

### Google Cloud Platform Setup

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Required APIs**:
   ```bash
   # Enable all required APIs
   gcloud services enable calendar.googleapis.com
   gcloud services enable gmail.googleapis.com
   gcloud services enable maps.googleapis.com
   gcloud services enable drive.googleapis.com
   gcloud services enable sheets.googleapis.com
   gcloud services enable tasks.googleapis.com
   gcloud services enable fitness.googleapis.com
   ```

3. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Set application type to "Desktop application"
   - Download the credentials JSON file

4. **Set Environment Variables**:
   ```bash
   export GOOGLE_CLIENT_ID="your-client-id"
   export GOOGLE_CLIENT_SECRET="your-client-secret"
   export GOOGLE_PROJECT_ID="your-project-id"
   ```

### Alternative Setup Methods

#### Run Backend Only
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Run Frontend Only
```bash
cd frontend
npm start
```

## 🧪 Testing

### Backend Tests
```bash
# Run all backend tests
cd backend
pytest

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests

# Run with coverage
pytest --cov=backend tests/
```

### Frontend Tests
```bash
cd frontend
npm test                    # Run tests
npm run test:ui            # Run tests with UI
```

### End-to-End Tests
```bash
# Run end-to-end tests from project root
pytest tests/
```

## 📚 API Documentation

### Core Endpoints

- `POST /api/chat` - Send messages to the AI personal trainer
- `GET /api/health` - Check backend health status
- `GET /api/docs` - Interactive API documentation (Swagger UI)

### Request/Response Examples

**Chat Request**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Schedule a workout for tomorrow at 6 PM"
    }
  ]
}
```

**Chat Response**:
```json
{
  "response": "I've scheduled a workout for tomorrow at 6 PM in your calendar. I've also sent you a reminder email with some workout suggestions based on your preferences."
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure backend is installed in development mode
   pip install -e backend/
   ```

2. **Google Authentication Issues**:
   ```bash
   # Re-run authentication setup
   python setup_auth.py
   ```

3. **Port Conflicts**:
   ```bash
   # Check for processes using ports 8000 or 3000
   lsof -i :8000
   lsof -i :3000
   ```

4. **Frontend Build Issues**:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

### Debug Mode

Enable debug logging by setting the log level in `run.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Check if services are running:
```bash
# Backend health
curl http://localhost:8000/api/health

# Frontend (should show React app)
curl http://localhost:3000
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest` and `npm test`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Add tests for new features
- Update documentation for API changes
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the backend
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- UI components from [Material-UI](https://mui.com/)
- Google APIs for service integration

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the API docs at `/api/docs`
- **Questions**: Open a discussion on GitHub

---

**Note**: This application requires Google API credentials and OpenAI API key to function. Make sure to follow the setup instructions carefully and keep your API keys secure. 