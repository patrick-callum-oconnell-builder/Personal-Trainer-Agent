# AI Personal Trainer

An AI-powered personal trainer application that helps users with their fitness goals through intelligent automation and Google service integration. The application can schedule workouts, manage meal planning, provide fitness advice, and automate various fitness-related tasks to save users time.

## 🎥 Demo

Here is a video demo of the functionality (in case you don't want to set up all of the Google Service API enabling, which isn't yet automated): 
https://drive.google.com/file/d/10pUME3WR3DRclbFq2YwOQmOTCEIaif9s/view?usp=sharing 

## ✨ Features

- **Real-time AI Conversation**: Chat with an intelligent personal trainer that understands context and provides personalized advice
- **Google Services Integration**: 
  - 📅 **Google Calendar**: Schedule and manage workouts automatically with conflict detection
  - 📧 **Gmail**: Send workout reminders and meal plans
  - 🗺️ **Google Maps**: Find nearby gyms, parks, and workout locations
  - 📁 **Google Drive**: Store and manage fitness documents and meal plans
  - 📊 **Google Sheets**: Track progress and maintain fitness logs
  - ✅ **Google Tasks**: Create and manage fitness-related tasks
  - 📱 **Google Fit**: Sync fitness data and track activities
- **Advanced Agent Orchestration**: State-based agent with intelligent decision-making for when to chat, use tools, or record preferences
- **Auto Tool Discovery**: Automatic discovery and registration of tools from Google services
- **Generic NLP Processing**: Unified natural language to structured argument conversion for all tools
- **Knowledge Graph**: Maintains context and user preferences across conversations
- **Modern Web Interface**: Clean React/MUI frontend with real-time chat and knowledge graph visualization
- **Conflict Detection**: Intelligent calendar conflict detection and resolution

## 🏗️ Architecture

### Tech Stack
- **Backend**: Python FastAPI with async/await support
- **Agent Framework**: Custom state-based orchestration with LangGraph-inspired patterns
- **AI/LLM**: OpenAI GPT models via LangChain
- **Frontend**: React 18 with Material-UI (MUI)
- **State Management**: Custom thread-safe state management with validation
- **Testing**: pytest with comprehensive unit and integration tests

### Project Structure
```
agent_personal_trainer/
├── backend/                    # FastAPI backend application
│   ├── agent_orchestration/    # Agent orchestration and state management
│   │   ├── agent_state_machine.py # State machine for agent workflow
│   │   ├── agent_state.py         # Thread-safe state management
│   │   ├── auto_tool_manager.py   # Automatic tool discovery and management
│   │   └── orchestrated_agent.py  # Main agent orchestration logic
│   ├── api/                   # API routes and endpoints
│   │   └── routes.py          # FastAPI router and endpoints
│   ├── google_services/       # Google API integrations
│   │   ├── auth.py            # Google authentication
│   │   ├── base.py            # Base service classes
│   │   ├── calendar.py        # Google Calendar integration
│   │   ├── gmail.py           # Gmail integration
│   │   ├── maps.py            # Google Maps integration
│   │   ├── drive.py           # Google Drive integration
│   │   ├── sheets.py          # Google Sheets integration
│   │   ├── tasks.py           # Google Tasks integration
│   │   └── fit.py             # Google Fit integration
│   ├── tools/                 # Agent tools and management
│   │   ├── personal_trainer_tool_manager.py # Main tool manager
│   │   ├── preferences_tools.py # User preference tools
│   │   └── tool_config.py     # Tool configuration and metadata
│   ├── utilities/             # Utility functions
│   │   ├── auth.py            # Authentication utilities
│   │   └── time_formatting.py # Time parsing utilities
│   ├── knowledge_graph.py     # Knowledge graph implementation
│   ├── prompts.py             # AI prompts and system messages
│   ├── personal_trainer_agent.py # Main agent class
│   ├── main.py                # FastAPI application entry point
│   └── tests/                 # Backend tests
│       ├── unit/              # Unit tests
│       └── integration/       # Integration tests
├── frontend/                  # React frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Chat.tsx       # Chat interface
│   │   │   ├── KnowledgeGraph.tsx # Knowledge graph visualization
│   │   │   └── StateHistory.tsx # State history display
│   │   └── App.tsx            # Main application component
│   └── package.json           # Frontend dependencies
├── tests/                     # End-to-end tests
├── run.py                     # Main application launcher
├── setup_auth.py              # Google authentication setup
└── README.md                  # This file
```

### Agent Architecture

The AI agent uses a sophisticated state machine with the following states:
- **Active**: Processing user input and making decisions
- **Awaiting User**: Waiting for user input
- **Awaiting Tool**: Executing a tool and waiting for results
- **Error**: Handling errors gracefully
- **Done**: Task completed

The agent can:
1. **Chat**: Provide fitness advice and answer questions
2. **Use Tools**: Execute Google service integrations with automatic discovery
3. **Record Preferences**: Store user preferences in the knowledge graph
4. **Manage State**: Maintain conversation context and user data
5. **Resolve Conflicts**: Handle calendar conflicts intelligently

### Auto Tool Discovery

The system features an advanced auto-discovery mechanism that:
- Automatically discovers tools from Google services
- Registers tools with metadata-based and reflection-based strategies
- Provides generic natural language to structured argument conversion
- Validates tool signatures and arguments
- Handles errors gracefully with detailed reporting

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

### Test Categories

The project includes several types of tests:

- **Unit Tests** (`backend/tests/unit/`) - Test individual components in isolation
- **Integration Tests** (`backend/tests/integration/`) - Test component interactions
- **End-to-End Tests** (`tests/`) - Test full application workflows

### Running Tests

#### All Tests
```bash
# Run all tests from project root
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=backend --cov-report=html
```

#### Backend Tests Only
```bash
# Run all backend tests
cd backend
pytest
```

#### End-to-End Tests
```bash
# Run all end-to-end tests
pytest tests/ -m e2e

# Run only fast end-to-end tests (exclude slow ones)
pytest tests/ -m e2e -m "not slow"

# Run specific end-to-end test
pytest tests/test_full_flow.py::test_basic_e2e
```

#### Test Markers
```bash
# Run tests by category
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m e2e               # End-to-end tests only
pytest -m slow              # Slow running tests
pytest -m google            # Tests requiring Google APIs

# Combine markers
pytest -m "e2e and not slow"  # Fast end-to-end tests only
```

### End-to-End Test Details

The end-to-end tests (`tests/`) verify complete application workflows:

- **`test_full_flow.py`** - Basic startup verification
- **`test_conversation_flow.py`** - Conversation and greeting flows
- **`test_advanced_flow.py`** - Complex workflows (calendar, sheets, etc.)

**Requirements for E2E Tests:**
- Chrome browser installed
- Google API credentials configured
- OpenAI API key set
- Ports 8000 and 3000 available

### Frontend Tests
```bash
cd frontend
npm test                    # Run tests
npm run test:ui            # Run tests with UI
```

## 📚 API Documentation

### Core Endpoints

- `POST /api/chat` - Send messages to the AI personal trainer
- `POST /api/chat/stream` - Stream responses from the AI personal trainer
- `GET /api/health` - Check backend health status
- `GET /api/docs` - Interactive API documentation (Swagger UI)
- `GET /api/knowledge-graph` - Get current knowledge graph state
- `GET /api/state-history` - Get agent state history
- `POST /api/state-history/clear` - Clear agent state history

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
  "response": "I've scheduled a workout for tomorrow at 6 PM in your calendar. I've also sent you a reminder email with some workout suggestions based on your preferences.",
  "type": "single"
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
- Powered by [LangChain](https://github.com/langchain-ai/langchain) for AI/LLM integration
- UI components from [Material-UI](https://mui.com/)
- Google APIs for service integration

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the API docs at `/api/docs`
- **Questions**: Open a discussion on GitHub

---

**Note**: This application requires Google API credentials and OpenAI API key to function. Make sure to follow the setup instructions carefully and keep your API keys secure. 