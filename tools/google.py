"""
Google service tools.

This module provides tools for interacting with various Google services through SerpAPI,
including search, finance, flights, and news.
"""

import json
import os
from tools.base import Tool


class GoogleSearchTool(Tool):
    """
    A tool that performs Google searches using the SerpAPI service.

    This tool enables language models to search the internet for real-time information
    by querying Google through SerpAPI. It returns search results including organic
    results, featured snippets, and other relevant search features.

    API key should be set in the SERPAPI_KEY environment variable.
    """

    def __init__(self):
        """
        Initialize the GoogleSearchTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines search options including
        the query, number of results, and optional location and language settings.
        """
        super().__init__(
            name="google_search",
            description="Search Google for information using SerpAPI",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to send to Google",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10, max: 100)",
                        "default": 10,
                    },
                    "location": {
                        "type": "string",
                        "description": "Location for geographically specific results (e.g., 'New York, New York, United States')",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for results (e.g., 'en' for English)",
                        "default": "en",
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, query, num_results=10, location=None, language="en"):
        """
        Performs a Google search using SerpAPI and returns the results.

        Makes a request to SerpAPI with the specified search parameters and returns
        a formatted version of the search results, including organic results,
        featured snippets, and other relevant search information.

        Args:
            query (str): The search query to send to Google
            num_results (int, optional): Number of results to return. Defaults to 10.
            location (str, optional): Location for geographically specific results. Defaults to None.
            language (str, optional): Language for results. Defaults to "en".

        Returns:
            str: JSON string containing search results or error information
        """
        try:
            import requests

            # Get API key from environment
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                print(
                    "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                )
                return json.dumps(
                    {
                        "error": "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                    }
                )

            # Build request parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": min(num_results, 100),  # Cap at 100 results
                "hl": language,
            }

            # Add optional location parameter
            if location:
                params["location"] = location

            # Make request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)

            # Check if request was successful
            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"SerpAPI request failed with status code {response.status_code}: {response.text}"
                    }
                )

            # Parse JSON response
            search_results = response.json()

            # Extract and format relevant information
            formatted_results = {
                "query": query,
                "organic_results": [],
                "answer_box": None,
                "knowledge_graph": None,
                "related_questions": [],
            }

            # Add organic results
            if "organic_results" in search_results:
                formatted_results["organic_results"] = [
                    {
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet"),
                        "position": result.get("position"),
                    }
                    for result in search_results["organic_results"][:num_results]
                ]

            # Add answer box if present
            if "answer_box" in search_results:
                answer_box = search_results["answer_box"]
                formatted_results["answer_box"] = {
                    "title": answer_box.get("title"),
                    "answer": answer_box.get("answer") or answer_box.get("snippet"),
                    "type": answer_box.get("type"),
                }

            # Add knowledge graph if present
            if "knowledge_graph" in search_results:
                kg = search_results["knowledge_graph"]
                formatted_results["knowledge_graph"] = {
                    "title": kg.get("title"),
                    "description": kg.get("description"),
                    "type": kg.get("type"),
                }

            # Add related questions if present
            if "related_questions" in search_results:
                formatted_results["related_questions"] = [
                    {
                        "question": q.get("question"),
                        "answer": q.get("answer"),
                    }
                    for q in search_results["related_questions"]
                ]

            return json.dumps(formatted_results, ensure_ascii=False)

        except ImportError:
            return json.dumps(
                {
                    "error": "Required package 'requests' is not installed. Install it with 'pip install requests'."
                }
            )
        except Exception as e:
            return json.dumps({"error": f"Error performing Google search: {str(e)}"})


class GoogleFinanceTool(Tool):
    """
    A tool that retrieves financial market data using SerpAPI's Google Finance endpoint.

    This tool enables language models to access real-time and historical stock market
    information including current prices, historical charts, key events, and related
    stocks. It's useful for financial analysis, market research, and tracking stocks.

    API key should be set in the SERPAPI_KEY environment variable.
    """

    def __init__(self):
        """
        Initialize the GoogleFinanceTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines options including
        the stock ticker query, time window for historical data, and language settings.
        """
        super().__init__(
            name="google_finance",
            description="Retrieve stock market and financial data using Google Finance",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol with exchange (e.g., 'AAPL:NASDAQ', 'GOOGL:NASDAQ', 'WMT:NYSE')",
                    },
                    "window": {
                        "type": "string",
                        "description": "Time window for historical data (e.g., '1D', '5D', '1M', '6M', '1Y', '5Y', 'MAX')",
                        "default": "1D",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for results (e.g., 'en' for English)",
                        "default": "en",
                    },
                },
                "required": ["ticker"],
            },
        )

    def execute(self, ticker, window="1D", language="en"):
        """
        Retrieves financial data for the specified stock ticker using SerpAPI.

        Makes a request to SerpAPI's Google Finance endpoint with the specified parameters
        and returns formatted stock information including price, market data, historical
        chart, and key events (if available).

        Args:
            ticker (str): Stock ticker symbol with exchange (e.g., 'AAPL:NASDAQ')
            window (str, optional): Time window for historical data. Defaults to "1D".
            language (str, optional): Language for results. Defaults to "en".

        Returns:
            str: JSON string containing financial data or error information
        """
        try:
            import requests

            # Get API key from environment
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                print(
                    "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                )
                return json.dumps(
                    {
                        "error": "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                    }
                )

            # Build request parameters
            params = {
                "engine": "google_finance",
                "q": ticker,
                "api_key": api_key,
                "hl": language,
            }

            # Add optional window parameter
            if window:
                params["window"] = window

            # Make request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)

            # Check if request was successful
            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"SerpAPI request failed with status code {response.status_code}: {response.text}"
                    }
                )

            # Parse JSON response
            finance_data = response.json()

            # Format the results (keeping all relevant data)
            formatted_data = {
                "ticker": ticker,
                "window": window,
            }

            # Add summary information if available
            if "summary" in finance_data:
                formatted_data["summary"] = finance_data["summary"]

            # Add key events if available (usually in 6M or longer windows)
            if "key_events" in finance_data:
                formatted_data["key_events"] = finance_data["key_events"]

            # Add graph data but limit points to avoid overwhelming responses
            if "graph" in finance_data:
                # Sample the graph data to avoid returning too many points
                graph_data = finance_data["graph"]

                # If very long graph, sample at regular intervals
                if len(graph_data) > 100:
                    sample_rate = len(graph_data) // 100
                    sampled_graph = graph_data[::sample_rate]
                else:
                    sampled_graph = graph_data

                formatted_data["graph"] = sampled_graph
                formatted_data["graph_points_count"] = len(sampled_graph)
                formatted_data["original_graph_points_count"] = len(graph_data)

            return json.dumps(formatted_data, ensure_ascii=False)

        except ImportError:
            return json.dumps(
                {
                    "error": "Required package 'requests' is not installed. Install it with 'pip install requests'."
                }
            )
        except Exception as e:
            return json.dumps({"error": f"Error retrieving finance data: {str(e)}"})


class GoogleFlightsTool(Tool):
    """
    A tool that searches for flight information using SerpAPI's Google Flights endpoint.

    This tool enables language models to access flight information including routes,
    prices, airlines, layovers, and flight durations. It's useful for travel planning,
    comparing flight options, and finding the best travel routes.

    API key should be set in the SERPAPI_KEY environment variable.
    """

    def __init__(self):
        """
        Initialize the GoogleFlightsTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines flight search options including
        departure/arrival locations, dates, and optional parameters like trip type and airlines.
        """
        super().__init__(
            name="google_flights",
            description="Search for flight information using Google Flights",
            parameters={
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "Departure location (city or airport code, e.g., 'New York' or 'JFK')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination location (city or airport code, e.g., 'Tokyo' or 'NRT')",
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format (e.g., '2024-10-15')",
                    },
                    "return_date": {
                        "type": "string",
                        "description": "Return date in YYYY-MM-DD format for round trips (e.g., '2024-10-25')",
                    },
                    "trip_type": {
                        "type": "string",
                        "description": "Type of trip: 'one-way', 'round-trip', or 'multi-city'",
                        "enum": ["one-way", "round-trip", "multi-city"],
                        "default": "round-trip",
                    },
                    "adults": {
                        "type": "integer",
                        "description": "Number of adult passengers",
                        "default": 1,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of flight options to return",
                        "default": 5,
                    },
                },
                "required": ["departure", "destination", "date"],
            },
        )

    def execute(
        self,
        departure,
        destination,
        date,
        return_date=None,
        trip_type="round-trip",
        adults=1,
        max_results=5,
    ):
        """
        Searches for flights using SerpAPI's Google Flights endpoint.

        Makes a request to SerpAPI with the specified flight parameters and returns
        formatted flight information including routes, prices, airlines, and duration.

        Args:
            departure (str): Departure location (city or airport code)
            destination (str): Destination location (city or airport code)
            date (str): Departure date in YYYY-MM-DD format
            return_date (str, optional): Return date for round trips. Defaults to None.
            trip_type (str, optional): Type of trip ('one-way', 'round-trip', 'multi-city'). Defaults to "round-trip".
            adults (int, optional): Number of adult passengers. Defaults to 1.
            max_results (int, optional): Maximum number of flight options to return. Defaults to 5.

        Returns:
            str: JSON string containing flight search results or error information
        """
        try:
            import requests

            # Get API key from environment
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                return json.dumps(
                    {
                        "error": "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                    }
                )

            # Build request parameters
            params = {
                "engine": "google_flights",
                "departure_id": departure,
                "arrival_id": destination,
                "outbound_date": date,
                "api_key": api_key,
                "hl": "en",
                "adults": adults,
            }

            # Add return date for round trips
            if trip_type == "round-trip" and return_date:
                params["return_date"] = return_date
            elif trip_type == "one-way":
                params["trip_type"] = "one-way"

            # Make request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)

            # Check if request was successful
            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"SerpAPI request failed with status code {response.status_code}: {response.text}"
                    }
                )

            # Parse JSON response
            flight_data = response.json()

            # Format the results for better readability
            formatted_results = {
                "departure": departure,
                "destination": destination,
                "date": date,
                "return_date": return_date,
                "trip_type": trip_type,
                "best_flights": [],
                "other_options": [],
                "airports": {},
            }

            # Extract best flights
            if "best_flights" in flight_data:
                for idx, flight in enumerate(flight_data["best_flights"]):
                    if idx >= max_results:
                        break

                    flight_info = {
                        "price": flight.get("price"),
                        "duration": flight.get("total_duration"),
                        "type": flight.get("type"),
                        "flights": [],
                    }

                    # Extract flight segments
                    if "flights" in flight:
                        for segment in flight["flights"]:
                            flight_info["flights"].append(
                                {
                                    "departure": {
                                        "airport": segment.get(
                                            "departure_airport", {}
                                        ).get("name"),
                                        "code": segment.get(
                                            "departure_airport", {}
                                        ).get("id"),
                                        "time": segment.get(
                                            "departure_airport", {}
                                        ).get("time"),
                                    },
                                    "arrival": {
                                        "airport": segment.get(
                                            "arrival_airport", {}
                                        ).get("name"),
                                        "code": segment.get("arrival_airport", {}).get(
                                            "id"
                                        ),
                                        "time": segment.get("arrival_airport", {}).get(
                                            "time"
                                        ),
                                    },
                                    "airline": segment.get("airline"),
                                }
                            )

                    # Extract layover information
                    if "layovers" in flight:
                        flight_info["layovers"] = [
                            {
                                "airport": layover.get("name"),
                                "code": layover.get("id"),
                                "duration": layover.get("duration"),
                            }
                            for layover in flight["layovers"]
                        ]

                    formatted_results["best_flights"].append(flight_info)

            # Extract other flight options
            if (
                "other_flights" in flight_data
                and len(formatted_results["best_flights"]) < max_results
            ):
                remaining = max_results - len(formatted_results["best_flights"])
                for idx, flight in enumerate(flight_data["other_flights"]):
                    if idx >= remaining:
                        break

                    flight_info = {
                        "price": flight.get("price"),
                        "duration": flight.get("total_duration"),
                        "type": flight.get("type"),
                        "flights": [],
                    }

                    # Extract flight segments
                    if "flights" in flight:
                        for segment in flight["flights"]:
                            flight_info["flights"].append(
                                {
                                    "departure": {
                                        "airport": segment.get(
                                            "departure_airport", {}
                                        ).get("name"),
                                        "code": segment.get(
                                            "departure_airport", {}
                                        ).get("id"),
                                        "time": segment.get(
                                            "departure_airport", {}
                                        ).get("time"),
                                    },
                                    "arrival": {
                                        "airport": segment.get(
                                            "arrival_airport", {}
                                        ).get("name"),
                                        "code": segment.get("arrival_airport", {}).get(
                                            "id"
                                        ),
                                        "time": segment.get("arrival_airport", {}).get(
                                            "time"
                                        ),
                                    },
                                    "airline": segment.get("airline"),
                                }
                            )

                    # Extract layover information
                    if "layovers" in flight:
                        flight_info["layovers"] = [
                            {
                                "airport": layover.get("name"),
                                "code": layover.get("id"),
                                "duration": layover.get("duration"),
                            }
                            for layover in flight["layovers"]
                        ]

                    formatted_results["other_options"].append(flight_info)

            # Extract airport information
            if "airports" in flight_data:
                airports_data = flight_data["airports"]
                if isinstance(airports_data, list) and len(airports_data) > 0:
                    for category in ["departure", "arrival"]:
                        if category in airports_data[0]:
                            formatted_results["airports"][category] = []
                            for airport in airports_data[0][category]:
                                formatted_results["airports"][category].append(
                                    {
                                        "name": airport.get("airport", {}).get("name"),
                                        "code": airport.get("airport", {}).get("id"),
                                        "city": airport.get("city"),
                                        "country": airport.get("country"),
                                    }
                                )

            return json.dumps(formatted_results, ensure_ascii=False)

        except ImportError:
            return json.dumps(
                {
                    "error": "Required package 'requests' is not installed. Install it with 'pip install requests'."
                }
            )
        except Exception as e:
            return json.dumps({"error": f"Error searching for flights: {str(e)}"})


class GoogleNewsTool(Tool):
    """
    A tool that searches for news articles using SerpAPI's Google News endpoint.

    This tool enables language models to access current news information from Google News
    by querying through SerpAPI. It returns news articles, top stories, and related topics
    based on search queries or specific topics.

    API key should be set in the SERPAPI_KEY environment variable.
    """

    def __init__(self):
        """
        Initialize the GoogleNewsTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines search options including
        the query, number of results, optional topic token, and language settings.
        """
        super().__init__(
            name="google_news",
            description="Search Google News for current news articles and topics",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for news articles (optional if topic_token is provided)",
                    },
                    "topic_token": {
                        "type": "string",
                        "description": "Topic token for specific news categories (optional if query is provided)",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10, max: 100)",
                        "default": 10,
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for results (e.g., 'en' for English)",
                        "default": "en",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code for geographically specific results (e.g., 'us' for United States)",
                        "default": "us",
                    },
                },
                "anyOf": [{"required": ["query"]}, {"required": ["topic_token"]}],
            },
        )

    def execute(
        self, query=None, topic_token=None, num_results=10, language="en", country="us"
    ):
        """
        Searches Google News using SerpAPI and returns the results.

        Makes a request to SerpAPI's Google News endpoint with the specified parameters
        and returns formatted news article information including headlines, sources,
        publication dates, and article snippets.

        Args:
            query (str, optional): The search query for news articles. Defaults to None.
            topic_token (str, optional): Topic token for specific news categories. Defaults to None.
            num_results (int, optional): Number of results to return. Defaults to 10.
            language (str, optional): Language for results. Defaults to "en".
            country (str, optional): Country code for geographically specific results. Defaults to "us".

        Returns:
            str: JSON string containing news search results or error information
        """
        try:
            import requests

            # Get API key from environment
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                return json.dumps(
                    {
                        "error": "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                    }
                )

            # Build request parameters
            params = {
                "engine": "google_news",
                "api_key": api_key,
                "gl": country,
                "hl": language,
            }

            # Add query or topic token (at least one is required)
            if query:
                params["q"] = query
            elif topic_token:
                params["topic_token"] = topic_token
            else:
                return json.dumps(
                    {"error": "Either query or topic_token must be provided"}
                )

            # Make request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)

            # Check if request was successful
            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"SerpAPI request failed with status code {response.status_code}: {response.text}"
                    }
                )

            # Parse JSON response
            news_data = response.json()

            # Format the results for better readability
            formatted_results = {
                "query": query,
                "topic_token": topic_token,
                "news_results": [],
                "related_topics": [],
            }

            # Extract news stories
            if "news_results" in news_data:
                for idx, article in enumerate(news_data["news_results"]):
                    if idx >= num_results:
                        break

                    formatted_results["news_results"].append(
                        {
                            "title": article.get("title"),
                            "link": article.get("link"),
                            "snippet": article.get("snippet"),
                            "source": article.get("source", {}).get("name"),
                            "published_date": article.get("date"),
                            "position": article.get("position"),
                            "thumbnail": article.get("thumbnail"),
                        }
                    )

            # Add top stories if present
            if (
                "top_stories" in news_data
                and len(formatted_results["news_results"]) < num_results
            ):
                remaining = num_results - len(formatted_results["news_results"])
                for idx, story in enumerate(news_data["top_stories"]):
                    if idx >= remaining:
                        break

                    formatted_results["news_results"].append(
                        {
                            "title": story.get("title"),
                            "link": story.get("link"),
                            "source": story.get("source"),
                            "published_date": story.get("date"),
                            "position": story.get("position"),
                            "thumbnail": story.get("thumbnail"),
                            "is_top_story": True,
                        }
                    )

            # Extract related topics
            if "related_topics" in news_data:
                for topic in news_data["related_topics"]:
                    formatted_results["related_topics"].append(
                        {
                            "title": topic.get("title"),
                            "topic_token": topic.get("topic_token"),
                            "thumbnail": topic.get("thumbnail"),
                        }
                    )

            return json.dumps(formatted_results, ensure_ascii=False)

        except ImportError:
            return json.dumps(
                {
                    "error": "Required package 'requests' is not installed. Install it with 'pip install requests'."
                }
            )
        except Exception as e:
            return json.dumps({"error": f"Error searching for news: {str(e)}"})
